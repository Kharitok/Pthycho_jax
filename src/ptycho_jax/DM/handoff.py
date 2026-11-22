"""Exact transfer to the untouched AD model, with scan positions frozen."""

import time
import jax
import jax.numpy as jnp
import numpy as np
from ptycho_jax.transmission.transmission_ptycho_models import (
    construct_point_model_ptycho_transmission,
)
from ptycho_jax.transmission.loss_and_reg import gauss_loss
from ptycho_jax.transmission.optimizers import prepare_optimizer
from ptycho_jax.transmission.optim_loop import run_optimization_loop
from .conventions import weights_to_a, a_to_weights_param
from .gauge import normalize_modes


def to_ad_parameters(state, geom, max_correction_magnitude=5.0):
    """DMState -> AD params and fixed arrays; no optimizer position leaf.

    Probe (K,Y,X) absorbs mean amplitude scale; matrices are (J,K,K). Arrays
    are copied so a later donated DM state cannot invalidate the naive AD start.
    """
    scale = jnp.sqrt(jnp.mean(state.a**2, axis=0))
    scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    params = {
        "sample": jnp.array(state.obj, copy=True),
        "probe_modes": state.probe * scale[:, None, None],
        "modal_weights": a_to_weights_param(state.a / scale[None]).astype(
            state.probe.dtype
        ),
    }
    fixed = {
        "sample_positions": jnp.array(geom.n, copy=True),
        "scan_mistakes": jnp.arctanh(-geom.delta / max_correction_magnitude),
    }
    return params, fixed


def from_ad_parameters(params):
    """Return equivalent unit probe (K,Y,X), object (H,W), a (J,K) for diagonal AD weights."""
    a = weights_to_a(params["modal_weights"])
    probe, a = normalize_modes(params["probe_modes"], a)
    return probe, params["sample"], a


def reference_point_model(geom, max_correction_magnitude=5.0):
    """Reference point model with fixed shift latents injected from fixed params."""
    original = construct_point_model_ptycho_transmission(
        {
            "probe_type": "fluctuating",
            "sample_type": "complex",
            "binning_factor": (1, 1),
            "sample_selector_type": "correcting",
            "propagator_type": "Fourrier",
            "probe_shape": geom.frame_shape,
            "shift_margin": geom.margin,
            "max_correction_magnitude": max_correction_magnitude,
        }
    )["point_model"]

    def point(params, fixed, index):
        with jax.default_matmul_precision("highest"):
            return original(
                {**params, "scan_mistakes": fixed["scan_mistakes"]}, fixed, index
            )

    return point


def evaluate_ad(params, fixed, geom, data, batch_size=13):
    """Evaluate reference intensities (J,Y,X), exact masked MSE and representative images."""
    predict = jax.jit(jax.vmap(reference_point_model(geom), in_axes=(None, None, 0)))
    losses = []
    totals = []
    first = None
    mean = np.zeros(geom.frame_shape, dtype=np.float64)
    root = np.asarray(data.sqrt_I)
    mask = np.broadcast_to(np.asarray(data.mask), root.shape)
    for start in range(0, len(root), batch_size):
        stop = min(start + batch_size, len(root))
        intensity = np.asarray(
            predict(params, fixed, jnp.arange(start, stop, dtype=jnp.int32))
        )
        residual = (np.sqrt(intensity + 1e-20) - root[start:stop]) ** 2
        m = mask[start:stop]
        losses.extend(
            np.sum(np.where(m, residual, 0), axis=(-2, -1), dtype=float)
            / np.sum(m, axis=(-2, -1))
        )
        totals.extend(np.sum(np.where(m, intensity, 0), axis=(-2, -1), dtype=float))
        mean += intensity.sum(axis=0, dtype=float)
        if first is None:
            first = intensity[0]
    return {
        "loss": float(np.mean(losses)),
        "frame_losses": np.asarray(losses),
        "predicted_totals": np.asarray(totals),
        "mean_prediction": mean / len(root),
        "first_prediction": first,
    }


def refine_ad(
    params,
    fixed,
    geom,
    data,
    *,
    n_steps=400,
    batch_size=13,
    seed=0,
    learning_rates=None,
):
    """Continue sample/probe/full shot matrices with optax; fixed positions never enter gradients."""
    rates = (
        {"sample": 0.003, "probe_modes": 0.03, "modal_weights": 0.0003}
        if learning_rates is None
        else learning_rates
    )
    initial = evaluate_ad(params, fixed, geom, data, batch_size)
    original = {k: np.asarray(v).copy() for k, v in fixed.items()}
    point = reference_point_model(geom)

    def batch_loss(params, fixed, index, measured, mask):
        predicted = jax.vmap(point, in_axes=(None, None, 0))(params, fixed, index)
        selected_mask = (
            mask[index] if mask.ndim == 3 else jnp.broadcast_to(mask, predicted.shape)
        )
        return jnp.mean(jax.vmap(gauss_loss)(predicted, measured, selected_mask))

    loss = jax.jit(batch_loss)
    loss_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0))
    optimizer = prepare_optimizer(
        {
            name: {"type": "adam", "learning_rate": value}
            for name, value in rates.items()
        }
    )
    start = time.perf_counter()
    params, fixed, _, history = run_optimization_loop(
        params=params,
        non_diff_params=fixed,
        opt_state=optimizer.init(params),
        optimizer=optimizer,
        loss_and_grad_fn=loss_and_grad,
        measured_batch_pool=np.asarray(data.sqrt_I),
        mask=data.mask,
        # Existing fused full-pass runner fixes its loss carry to float32.
        # Its streaming variant supports independent complex128 algebra trials.
        mode="accumulate_full_pass"
        if params["sample"].dtype == jnp.complex64 and not jax.config.jax_enable_x64
        else "accumulate_full_pass_streaming",
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        shuffle_each_epoch=True,
        use_multigpu=False,
    )
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - start
    for key, value in original.items():
        if not np.array_equal(value, np.asarray(fixed[key])):
            raise RuntimeError("Fixed scan geometry changed during AD.")
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(params)):
        raise FloatingPointError("AD produced nonfinite parameters.")
    final = evaluate_ad(params, fixed, geom, data, batch_size)
    return (
        params,
        np.asarray(history),
        {
            "initial_loss": initial["loss"],
            "final_loss": final["loss"],
            "optimization_seconds": elapsed,
            "positions_frozen": True,
            "learning_rates": rates,
        },
    )
