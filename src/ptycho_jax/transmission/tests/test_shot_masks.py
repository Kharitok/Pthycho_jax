"""Shot masks must select their own valid pixels through the real AD pipeline."""

import json
from dataclasses import replace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import pipeline_nanomax_upd as base
import pipeline_nanomax_upd_v2 as pipeline
from ptycho_jax.transmission.background import (
    audit_background,
    background_components,
    initialize_background,
    shared_scaled_background,
)
from ptycho_jax.transmission.loss_and_reg import (
    gauss_loss,
    join_loss_and_forward_batched,
)
from ptycho_jax.transmission.optim_loop import run_optimization_loop


def test_batched_loss_selects_masks_by_shuffled_frame_id_and_normalizes_each_shot():
    masks = jnp.array([[[1, 0], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [0, 0]]], bool)
    roots = jnp.array([[[1, 100], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [100, 100]]])
    loss = join_loss_and_forward_batched(
        gauss_loss, lambda p, f, i: jnp.full((2, 2), p["intensity"])
    )
    indices = jnp.array([2, 0])
    value, grad = jax.jit(jax.value_and_grad(loss))(
        {"intensity": jnp.array(4.0)}, {}, indices, roots[indices], masks
    )
    assert float(value) == pytest.approx(1.0)
    assert float(grad["intensity"]) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    "mode",
    [
        "accumulate_full_pass",
        "accumulate_full_pass_streaming",
        "sequential_no_repeats",
        "random_with_replacement",
    ],
)
def test_real_optimizer_ignores_bad_pixels_with_shot_masks(mode):
    roots = np.array(
        [
            [[1.0, 100.0], [1.0, 1.0]],
            [[100.0, 2.0], [100.0, 100.0]],
            [[3.0, 3.0], [100.0, 3.0]],
        ],
        dtype=np.float32,
    )
    mask = jnp.asarray(roots < 100)
    loss = join_loss_and_forward_batched(
        gauss_loss, lambda p, f, i: jnp.full((2, 2), p["amplitude"][i] ** 2)
    )
    params = {"amplitude": jnp.array([0.5, 0.5, 0.5])}
    optimizer = optax.sgd(0.1)
    fitted, _, _, history = run_optimization_loop(
        params=params,
        non_diff_params={},
        opt_state=optimizer.init(params),
        optimizer=optimizer,
        loss_and_grad_fn=jax.jit(jax.value_and_grad(loss)),
        measured_batch_pool=roots,
        mask=mask,
        mode=mode,
        n_steps=8,
        batch_size=2,
        seed=1,
    )
    assert np.isfinite(history).all()
    values = np.asarray(fitted["amplitude"])
    assert np.all(values >= 0.5) and np.all(values < [1.01, 2.01, 3.01])
    assert values[2] > 0.5


def test_evaluation_uses_each_shots_mask_for_loss_totals_and_mean():
    masks = np.array([[[1, 0], [1, 1]], [[1, 1], [0, 0]]], bool)
    predictions = jnp.array([[[4.0, 999.0], [4.0, 4.0]], [[4.0, 4.0], [999.0, 999.0]]])
    state = {
        "mask": masks,
        "root_measured": np.array([[[1, 0], [1, 1]], [[3, 3], [0, 0]]]),
        "predict_batch": lambda p, f, i: predictions[i],
    }
    result = base.evaluate_reconstruction({}, {}, state, batch_size=1)
    np.testing.assert_allclose(result["frame_losses"], [1.0, 1.0])
    np.testing.assert_allclose(result["predicted_totals"], [12.0, 8.0])
    # Means represent masked counts per frame, so background audits share support.
    np.testing.assert_allclose(result["mean_prediction"], [[4.0, 2.0], [2.0, 2.0]])


def test_background_audit_counts_only_pixels_observed_in_each_shot():
    masks = np.array([[[1, 0], [1, 1]], [[1, 1], [0, 0]]], bool)
    params, fixed = initialize_background(
        masks.any(axis=0),
        np.array([1000.0, 2000.0]),
        (2, 2),
        0.02,
        0.1,
        shot_masks=masks,
    )
    output = jax.vmap(shared_scaled_background, in_axes=(None, None, 0))(
        params, fixed, jnp.arange(2)
    )
    # Shared full map is 5 and 10 counts/pixel; only 3 and 2 pixels are observed.
    np.testing.assert_allclose(
        np.asarray(output).sum(axis=(1, 2)), [20.0, 40.0], rtol=1e-6
    )
    c = background_components(params, fixed)
    np.testing.assert_allclose(c["observed_frame_counts"], [15.0, 20.0], rtol=1e-6)
    np.testing.assert_allclose(c["fractions"], [0.015, 0.01], rtol=1e-6)
    np.testing.assert_allclose(
        c["mean_observed_map"], [[7.5, 5.0], [2.5, 2.5]], rtol=1e-6
    )
    _, audit = audit_background(
        params,
        fixed,
        np.full((2, 2), 750.0),
        np.full((2, 2), 750.0),
        np.array([1000.0, 2000.0]),
    )
    assert audit["mean_measured_fraction"] == pytest.approx(35 / 3000, rel=1e-6)
    assert audit["max_frame_measured_fraction"] == pytest.approx(0.015, rel=1e-6)
    assert audit["mean_counts_per_valid_pixel"] == pytest.approx(7.0, rel=1e-6)


def test_background_budget_saturation_uses_full_shared_map_with_missing_pixels():
    masks = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]], bool)
    params, fixed = initialize_background(
        np.ones((2, 2), bool),
        np.array([1000.0, 1000.0]),
        (2, 2),
        0.02,
        0.1,
        shot_masks=masks,
    )
    params["background_scale_logits"] = jnp.full(2, 20.0)
    _, audit = audit_background(
        params,
        fixed,
        np.full((2, 2), 500.0),
        np.full((2, 2), 500.0),
        np.array([1000.0, 1000.0]),
    )
    assert audit["max_frame_measured_fraction"] == pytest.approx(0.05, rel=1e-6)
    assert audit["fraction_frames_near_budget"] == 1.0


@pytest.fixture
def shot_config(tmp_path):
    y, x = np.indices((24, 24))
    image = (500 * np.exp(-((y - 12) ** 2 + (x - 12) ** 2) / 14) + 1).astype(np.uint32)
    frames = np.stack([image, 2 * image, 3 * image])
    for i, (row, col) in enumerate([(9, 10), (10, 11), (11, 12)]):
        frames[i, row, col] = np.iinfo(np.uint32).max
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as f:
        f["/entry/measurement/eiger4m/frames"] = frames
        f["/entry/measurement/sx"] = [0.1, -0.1, 0]
        f["/entry/measurement/sy"] = [-0.1, 0.1, 0]
        f["/entry/snapshots/pre_scan/energy_raw"] = [10000.0]
    settings = {
        "datapath": str(path),
        "output_dir": str(tmp_path / "out"),
        "tag": "shot_masks",
        "n_steps": 4,
        "batch_size": 2,
        "crop_shape": [16, 16],
        "crop_center": [12, 12],
        "shift_margin": 3,
        "max_correction_magnitude": 2.0,
        "object_padding": 4,
        "background": True,
        "background_grid_shape": [4, 4],
        "background_start_step": 0,
        "diagnostics": True,
        "z_samples": 5,
        "padding_factor": 1,
        "shot_specific_masks": True,
        "max_positions": 2,
    }
    json_path = tmp_path / "settings.json"
    json_path.write_text(json.dumps(settings))
    return pipeline.parse_configuration(["--config", str(json_path)])[0]


@pytest.mark.parametrize("background", [False, True])
def test_json_pipeline_preserves_shot_masks_and_initial_energy_and_saves_them(
    shot_config, background
):
    shot_config = replace(shot_config, background=background)
    data = base.load_reconstruction_data(shot_config)
    assert data["shot_masks"].shape == (2, 16, 16)
    assert data["shot_masks"].dtype == bool
    # Reconstruction sorting selects acquisition frames 1,2; vertical flip then FFT shift.
    np.testing.assert_array_equal(data["frame_indices"], [1, 2])
    expected = np.ones((2, 16, 16), bool)
    expected[0, 1, 15] = expected[1, 0, 0] = False
    np.testing.assert_array_equal(data["shot_masks"], expected)
    assert data["measured_intensities"].max() < 2000
    state = base.initialize_reconstruction(data, shot_config, 2)
    np.testing.assert_array_equal(state["mask"], expected)
    initial = base.evaluate_reconstruction(state["params"], state["fixed"], state, 2)
    np.testing.assert_allclose(
        initial["predicted_totals"], data["total_ints"], rtol=2e-5
    )
    result = pipeline.run_pipeline(shot_config, [2])[0]
    assert result["final_loss"] < result["initial_loss"]
    assert result["shot_specific_masks"] is True
    if background:
        assert result["background"]["max_frame_measured_fraction"] <= 0.05
    folder = shot_config.run_dir / "modes_2"
    with np.load(shot_config.artifact("reconstruction.npz", folder)) as saved:
        np.testing.assert_array_equal(saved["shot_masks"], expected)
        np.testing.assert_array_equal(saved["detector_mask"], expected.any(axis=0))
        assert np.isfinite(saved["sample"]).all()
        if background:
            assert saved["background_observed_frame_counts"].shape == (2,)
    assert shot_config.artifact("results.md", folder / "diagnostics").is_file()


def test_cli_can_disable_shot_masks_from_json(shot_config):
    config, _, _, _ = pipeline.parse_configuration(["--shot-specific-masks"])
    assert config.shot_specific_masks is True
    config, _, _, _ = pipeline.parse_configuration(
        [
            "--config",
            str(shot_config.datapath.parent / "settings.json"),
            "--no-shot-specific-masks",
        ]
    )
    assert config.shot_specific_masks is False
