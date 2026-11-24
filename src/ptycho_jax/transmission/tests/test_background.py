"""Detector backgrounds must be additive, identifiable and photon-budget limited."""

import importlib
import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import pipeline_nanomax_upd as base
import pipeline_nanomax_upd_v2 as pipeline
from ptycho_jax.transmission import loss_and_reg
from ptycho_jax.transmission import transmission_ptycho_models as models


def background_module():
    return importlib.import_module("ptycho_jax.transmission.background")


def test_shared_background_scaling_mask_budget_and_canonical_representation():
    bg = background_module()
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    params, fixed = bg.initialize_background(
        mask, np.array([1000.0, 2000.0]), (2, 2), 0.02, 0.1
    )
    output = jax.jit(jax.vmap(bg.shared_scaled_background, in_axes=(None, None, 0)))(
        params, fixed, jnp.arange(2)
    )
    np.testing.assert_allclose(
        np.asarray(output).sum(axis=(1, 2)), [20.0, 40.0], rtol=1e-6
    )
    np.testing.assert_array_equal(output[:, 0, 0], 0)
    np.testing.assert_allclose(output[1], 2 * output[0], rtol=1e-6)
    canonical = bg.background_components(params, fixed)
    np.testing.assert_allclose(canonical["map"].sum(), 30.0, rtol=1e-6)
    np.testing.assert_allclose(canonical["scales"], [2 / 3, 4 / 3], rtol=1e-6)
    np.testing.assert_allclose(
        canonical["scales"][:, None, None] * canonical["map"], output, rtol=1e-6
    )
    extreme = {**params, "background_scale_logits": jnp.array([-1000.0, 1000.0])}
    bounded = jax.vmap(bg.shared_scaled_background, in_axes=(None, None, 0))(
        extreme, fixed, jnp.arange(2)
    )
    assert np.isfinite(bounded).all() and np.min(bounded) >= 0
    np.testing.assert_allclose(
        np.asarray(bounded).sum(axis=(1, 2)), [0.0, 200.0], atol=1e-4
    )


def test_background_is_added_after_binning_and_before_detector_response():
    bg = background_module()
    params, fixed = bg.initialize_background(
        np.ones((2, 2), bool), np.array([1000.0]), (2, 2), 0.02, 0.1
    )
    detector = models.prepare_detector(
        models.intensifier,
        models.prepare_detector_binning((2, 2)),
        lambda intensity, p, f, i: intensity**2,
        background=bg.shared_scaled_background,
    )
    # Unit field -> four counts in each 2x2 bin; background=20/4=5 per bin.
    out = detector(jnp.ones((1, 4, 4), dtype=jnp.complex64), params, fixed, 0)
    np.testing.assert_allclose(out, np.full((2, 2), 81.0), atol=1e-5)
    old_detector = models.prepare_detector(
        models.intensifier,
        models.prepare_detector_binning((2, 2)),
        models.identity_non_linearity,
    )
    np.testing.assert_array_equal(
        old_detector(jnp.ones((1, 4, 4)), {}, {}, 0), np.full((2, 2), 4.0)
    )


def test_background_detector_factory_and_gradients_are_real_and_finite():
    bg = background_module()
    params, fixed = bg.initialize_background(
        np.ones((6, 6), bool), np.array([1000.0, 1500.0]), (2, 2), 0.01, 0.1
    )
    params.update(
        sample=jnp.ones((8, 8), dtype=jnp.complex64),
        probe_modes=jnp.ones((1, 6, 6), dtype=jnp.complex64),
    )
    fixed["sample_positions"] = jnp.zeros((2, 2), dtype=jnp.int32)
    model = models.construct_point_model_ptycho_transmission(
        {
            "probe_type": "static",
            "sample_type": "complex",
            "probe_shape": (6, 6),
            "shift_margin": 0,
            "background_type": "shared_scaled",
        }
    )
    expected_signal = np.zeros((6, 6))
    expected_signal[0, 0] = 36.0
    prediction = jax.jit(model["point_model"])(params, fixed, 1)
    np.testing.assert_allclose(prediction, expected_signal + 15 / 36, atol=2e-5)
    value, gradients = jax.value_and_grad(
        lambda p: jnp.sum(model["point_model"](p, fixed, 0))
    )(params)
    assert np.isfinite(value)
    assert np.isfinite(gradients["background_logits"]).all()
    assert np.isfinite(gradients["background_scale_logits"]).all()
    assert gradients["background_scale_logits"][0] > 0
    assert gradients["background_scale_logits"][1] == 0


def test_background_regularizer_penalizes_amount_structure_and_scale_jitter():
    bg = background_module()
    params, fixed = bg.initialize_background(
        np.ones((4, 4), bool), np.array([1000.0, 1000.0]), (2, 2), 0.02, 0.1
    )
    reg = loss_and_reg.create_background_regularizer(
        l1_weight=2.0, smoothness_weight=3.0, scale_weight=4.0
    )
    # Uniform map has no smoothness cost; equal levels have no scale cost.
    assert float(reg(params, fixed)) == pytest.approx(2 * 20 / 16, rel=1e-6)
    structured = {
        **params,
        "background_logits": params["background_logits"].at[0, 0].set(2.0),
    }
    assert reg(structured, fixed) > reg(params, fixed)
    jitter = {
        **params,
        "background_scale_logits": params["background_scale_logits"]
        + jnp.array([-1.0, 1.0]),
    }
    scale_only = loss_and_reg.create_background_regularizer(0.0, 0.0, 1.0)
    assert scale_only(jitter, fixed) > scale_only(params, fixed)
    assert all(
        np.isfinite(x).all() for x in jax.tree.leaves(jax.grad(reg)(params, fixed))
    )


def test_known_additive_background_is_recovered_without_changing_signal():
    bg = background_module()
    yy, xx = np.indices((8, 8))
    signal = np.stack(
        [20 + 2 * np.cos(xx), 25 + 3 * np.sin(yy), 30 + np.cos(xx + yy)]
    ).astype(np.float32)
    fractions = np.array([0.005, 0.01, 0.02])
    totals = signal.sum(axis=(1, 2), dtype=float) / (1 - fractions)
    truth_logits = 0.25 * np.cos(xx / 4) + 0.2 * np.sin(yy / 3)
    probability = np.exp(truth_logits)
    probability /= probability.sum()
    counts = totals * fractions
    target = signal + counts[:, None, None] * probability
    params, fixed = bg.initialize_background(
        np.ones((8, 8), bool), totals, (8, 8), 0.001, 0.05
    )
    optimizer = optax.adam(0.08)

    @jax.jit
    def step(p, s):
        def loss(q):
            background = jax.vmap(bg.shared_scaled_background, in_axes=(None, None, 0))(
                q, fixed, jnp.arange(3)
            )
            return jnp.mean((jnp.sqrt(signal + background) - jnp.sqrt(target)) ** 2)

        value, gradient = jax.value_and_grad(loss)(p)
        updates, s = optimizer.update(gradient, s, p)
        return optax.apply_updates(p, updates), s, value

    state = optimizer.init(params)
    for _ in range(500):
        params, state, value = step(params, state)
    recovered = bg.background_components(params, fixed)
    assert float(value) < 1e-8
    np.testing.assert_allclose(recovered["frame_counts"], counts, rtol=0.005)
    np.testing.assert_allclose(
        recovered["probability"], probability, rtol=0.01, atol=1e-4
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"background_max_fraction": 0},
        {"background_max_fraction": 1},
        {"background_initial_fraction": 0},
        {"background_initial_fraction": 0.2},
        {"background_grid_shape": (1, 2)},
        {"background_start_step": -1},
        {"background_learning_rate": -1},
        {"background_l1_weight": -1},
    ],
)
def test_invalid_background_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        pipeline.ReconstructionConfig(**kwargs)


@pytest.mark.parametrize("start_step", [0, 10])
def test_background_pipeline_initial_energy_and_saved_dominance_checks(
    tmp_path, start_step
):
    y, x = np.indices((24, 24))
    image = (500 * np.exp(-((y - 12) ** 2 + (x - 12) ** 2) / 14) + 1).astype(np.uint32)
    frames = np.stack([image, 2 * image, 3 * image])
    frames[:, 0, 0] = np.iinfo(np.uint32).max
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f["/entry/measurement/eiger4m/frames"] = frames
        f["/entry/measurement/sx"] = [-0.1, 0, 0.1]
        f["/entry/measurement/sy"] = [-0.1, 0.1, 0]
        f["/entry/snapshots/pre_scan/energy_raw"] = [10000.0]
    config = pipeline.ReconstructionConfig(
        datapath=path,
        output_dir=tmp_path / "out",
        tag="background",
        n_steps=4,
        batch_size=2,
        crop_shape=(16, 16),
        crop_center=(12, 12),
        shift_margin=3,
        max_correction_magnitude=2.0,
        object_padding=4,
        background=True,
        background_grid_shape=(4, 4),
        background_start_step=start_step,
        diagnostics=False,
    )
    data = base.load_reconstruction_data(config)
    state = base.initialize_reconstruction(data, config, 2)
    initial = base.evaluate_reconstruction(
        state["params"], state["fixed"], state, config.batch_size
    )
    np.testing.assert_allclose(
        initial["predicted_totals"], data["total_ints"], rtol=2e-5
    )
    result = pipeline.run_pipeline(config, [2])[0]
    audit = result["background"]
    assert audit["enabled"] and not audit["dominates_signal"]
    assert 0 < audit["max_frame_measured_fraction"] <= config.background_max_fraction
    folder = config.run_dir / "modes_2"
    with np.load(config.artifact("reconstruction.npz", folder)) as saved:
        assert saved["background_map"].shape == (16, 16)
        assert saved["background_scales"].shape == (3,)
        assert saved["background_logits"].shape == (4, 4)
        assert np.isfinite(saved["background_map"]).all()
        if start_step == 0:
            assert (
                np.max(
                    np.abs(
                        saved["background_scale_logits"]
                        - saved["initial_background_scale_logits"]
                    )
                )
                > 0
            )
        else:
            np.testing.assert_array_equal(
                saved["background_scale_logits"],
                saved["initial_background_scale_logits"],
            )
            np.testing.assert_array_equal(saved["background_logits"], np.zeros((4, 4)))
    assert config.artifact("background.png", folder).is_file()
    assert config.artifact("background_scaling.png", folder).is_file()
    assert json.loads(config.artifact("metrics.json", folder).read_text())[
        "background"
    ]["enabled"]
