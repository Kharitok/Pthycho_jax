"""Difference-map iterations; no gradient descent is used in this module."""

from functools import lru_cache
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .types import DMState
from .projections import fourier_projection, overlap_projection
from .operators import coverage_map
from .metrics import amplitude_metrics
from .gauge import active_modes


def schedule(it, cfg):
    """Iteration scalar -> traced probe boolean and weight-mode scalar 0/1/2."""
    probe = it >= cfg.probe_start
    if cfg.weight_strategy == "staged":
        mode = jnp.where(
            it >= cfg.per_position_weights_start,
            2,
            jnp.where(it >= cfg.global_weights_start, 1, 0),
        )
    else:
        mode = jnp.asarray(
            {"fixed": 0, "global": 1, "per_position": 2}[cfg.weight_strategy], jnp.int32
        )
    return probe, mode


def _project(psi, state, geom, cfg, covered):
    update_probe, weight_mode = schedule(state.it, cfg)
    return overlap_projection(
        psi,
        state.probe,
        state.obj,
        state.a,
        geom,
        cfg,
        update_probe,
        weight_mode,
        covered,
    )


def _finish(state, psi_new, psi_F, probe, obj, a, data, fstats, cfg):
    new = DMState(psi_new, probe, obj, a, state.it + 1)
    model = probe[None] * a[:, :, None, None]
    # Compute overlap fields from the returned P/O/a in the caller, not psi_new.
    update = jnp.sum(abs(psi_new - state.psi) ** 2) / jnp.maximum(
        jnp.sum(abs(psi_F) ** 2), jnp.finfo(a.dtype).tiny
    )
    stats = {
        **fstats,
        "update_norm": update,
        "weight_mode": schedule(state.it, cfg)[1],
        "weight_mean": jnp.mean(a * a, axis=0),
        "weight_std": jnp.std(a * a, axis=0),
        "weight_min": jnp.min(a, axis=0),
        "clamped_fraction": jnp.mean(a == 0),
        "inactive_modes": jnp.sum(~active_modes(a, cfg.collapse_threshold)),
        "probe_norms": jnp.sqrt(jnp.sum(abs(probe) ** 2, axis=(-2, -1))),
    }
    return new, stats


def dm_step(state, data, geom, cfg):
    """One beta=1 (or configured general-beta) step on a DMState pytree."""
    if cfg.beta != 1:
        return dm_step_general(state, data, geom, cfg)
    covered = coverage_map(geom, state.obj.real.dtype) > 0
    eps = cfg.fourier_eps_relative * jnp.sqrt(jnp.mean(data.sqrt_I**2))
    pf, stats = fourier_projection(state.psi, data.sqrt_I, data.mask, eps)
    overlap, p, o, a = _project(2 * pf - state.psi, state, geom, cfg, covered)
    new, metrics = _finish(
        state, state.psi + overlap - pf, pf, p, o, a, data, stats, cfg
    )
    metrics.update(amplitude_metrics(overlap, data))
    return new, metrics


def dm_step_general(state, data, geom, cfg):
    """General beta formula; second overlap call is stateless, even at beta=1."""
    beta = cfg.beta
    covered = coverage_map(geom, state.obj.real.dtype) > 0
    eps = cfg.fourier_eps_relative * jnp.sqrt(jnp.mean(data.sqrt_I**2))
    pf, stats = fourier_projection(state.psi, data.sqrt_I, data.mask, eps)
    overlap, p, o, a = _project(
        (1 + 1 / beta) * pf - state.psi / beta, state, geom, cfg, covered
    )
    po, *_ = _project(state.psi, state, geom, cfg, covered)
    reflected = (1 - 1 / beta) * po + state.psi / beta
    pf_other, _ = fourier_projection(reflected, data.sqrt_I, data.mask, eps)
    new, metrics = _finish(
        state, state.psi + beta * (overlap - pf_other), pf, p, o, a, data, stats, cfg
    )
    metrics.update(amplitude_metrics(overlap, data))
    return new, metrics


@lru_cache(maxsize=32)
def _block_function(cfg, count, donate):
    """Compile a scan block once; all large arrays remain explicit arguments."""

    def block(state, data, geom):
        return jax.lax.scan(
            lambda s, _: dm_step(s, data, geom, cfg), state, None, length=count
        )

    return jax.jit(block, donate_argnums=(0,) if donate else ())


def run_dm(
    state,
    data,
    geom,
    cfg,
    n_iter,
    *,
    log_every=10,
    donate=True,
    callback=None,
    debug_dir=None,
):
    """Run scan blocks, return final DMState and per-iteration NumPy metrics.

    A callback receives (state, history_block) after each block. Input state is
    donated by default and must not be read after the run.
    """
    history = {}
    for start in range(0, n_iter, log_every):
        count = min(log_every, n_iter - start)
        state, block = _block_function(cfg, count, donate)(state, data, geom)
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(state))
        if not finite:
            if debug_dir is not None:
                folder = Path(debug_dir)
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    folder / "dm_nonfinite.npz",
                    probe=np.asarray(state.probe),
                    obj=np.asarray(state.obj),
                    a=np.asarray(state.a),
                    it=np.asarray(state.it),
                )
            raise FloatingPointError(
                "DM generated nonfinite parameters; see debug snapshot if enabled."
            )
        block = jax.device_get(block)
        for name, values in block.items():
            history.setdefault(name, []).append(np.asarray(values))
        print(
            f"DM {int(state.it):4d}: model MSE={float(block['model_mse'][-1]):.6g}, "
            f"R={float(block['model_r'][-1]):.4g}, update={float(block['update_norm'][-1]):.4g}",
            flush=True,
        )
        if callback is not None:
            callback(state, block)
    return state, {name: np.concatenate(values) for name, values in history.items()}
