"""Numerical and end-to-end checks for constrained NanoMAX AD."""

import hashlib
import importlib
import json
import re
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ptycho_jax.transmission import loss_and_reg, projectors
from ptycho_jax.transmission.optim_loop import run_optimization_loop


def test_run_identity_is_shared_by_folder_and_files(tmp_path):
    pipeline = importlib.import_module("pipeline_nanomax_upd_v2")
    config = pipeline.ReconstructionConfig(
        output_dir=tmp_path,
        tag="projection test",
        started_utc="2026-11-23T12:34:56+00:00",
    )
    assert config.run_dir.name == "scan_000082_20261123T123456Z_projection_test"
    assert (
        config.artifact("sample.png").name
        == "20261123T123456Z_projection_test_sample.png"
    )
    assert config.tag == "projection test"
    assert config.artifact("sample.png").parent == config.run_dir
    assert config.artifact("sample.png") == config.artifact("sample.png")
    generated = pipeline.ReconstructionConfig(output_dir=tmp_path)
    assert re.fullmatch(r"[A-Za-z]{5}", generated.tag)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_tv_weight": -1},
        {"probe_tv_weight": float("nan")},
        {"tag": "../escape"},
        {"tag": "   "},
        {"probe_support_radius_px": 0},
        {"probe_support_decay_px": -1},
        {"started_utc": "2026-11-23T12:34:56"},
    ],
)
def test_invalid_run_configuration_is_rejected(kwargs):
    pipeline = importlib.import_module("pipeline_nanomax_upd_v2")
    with pytest.raises(ValueError):
        pipeline.ReconstructionConfig(**kwargs)


def test_parameter_tv_penalty_and_complex_gradient_match_independent_reference():
    # 2x2 complex ramp: squared horizontal/vertical differences are 4 and 9.
    params = {"sample": jnp.array([[0, 2j], [3, 3 + 2j]], dtype=jnp.complex64)}
    reg = loss_and_reg.create_parameter_tv_regularizer({"sample": 2.5}, epsilon=1e-3)
    value, grad = jax.jit(jax.value_and_grad(reg))(params, {})
    assert float(value) == pytest.approx(2.5 * np.sqrt(13 + 2e-6), rel=2e-6)
    assert np.isfinite(np.asarray(grad["sample"])).all()
    flat = {"sample": jnp.ones((3, 4), dtype=jnp.complex64)}
    assert np.isfinite(np.asarray(jax.grad(reg)(flat, {})["sample"])).all()
    eps = 1e-3
    perturb = jnp.zeros((2, 2), dtype=jnp.complex64).at[0, 1].set(1j * eps)
    numerical = (
        reg({"sample": params["sample"] + perturb}, {})
        - reg({"sample": params["sample"] - perturb}, {})
    ) / (2 * eps)
    assert float(numerical) == pytest.approx(
        -float(grad["sample"][0, 1].imag), abs=1e-3
    )


def test_regularizer_contributes_once_with_a_partial_final_batch():
    # A batch-independent penalty must retain its weight at batch sizes 2 and 3.
    reg = loss_and_reg.create_parameter_tv_regularizer({"sample": 3.0})
    data_loss = loss_and_reg.join_loss_and_forward_batched(
        loss_and_reg.gauss_loss, lambda p, f, i: jnp.ones((2, 2))
    )
    loss = jax.jit(jax.value_and_grad(loss_and_reg.join_loss_and_reg(data_loss, reg)))
    params = {"sample": jnp.array([[0.0, 1.0], [2.0, 3.0]])}
    expected = float(reg(params, {}))
    for size in (2, 3):
        optimizer = optax.sgd(0.0)
        _, _, _, history = run_optimization_loop(
            params,
            optimizer.init(params),
            optimizer,
            loss,
            {},
            np.ones((5, 2, 2), dtype=np.float32),
            jnp.ones((2, 2), dtype=bool),
            "accumulate_full_pass",
            1,
            size,
        )
        assert float(history[0]) == pytest.approx(expected, rel=1e-6)


def test_normalized_tv_cannot_be_reduced_by_compensated_amplitude_rescaling():
    params = {
        "sample": jnp.array([[1, 2j], [3, 3 + 2j]], dtype=jnp.complex64),
        "probe_modes": jnp.array([[[2j, 1], [1j, 2]], [[1j, 2j], [2, 1]]]),
        "modal_weights": jnp.ones(2),
    }
    reg = loss_and_reg.create_parameter_tv_regularizer(
        {"sample": 1.0, "probe_modes": 0.2}, normalize=True
    )
    expected = float(reg(params, {}))
    intensity = lambda p: jnp.sum(
        jnp.abs(
            jnp.fft.fft2(
                p["sample"] * p["probe_modes"] * p["modal_weights"][:, None, None]
            )
        )
        ** 2,
        axis=0,
    )
    for scale in (0.125, 8.0):
        scaled = {
            "sample": params["sample"] * scale,
            "probe_modes": params["probe_modes"] * scale,
            "modal_weights": params["modal_weights"] / scale**2,
        }
        np.testing.assert_allclose(intensity(scaled), intensity(params), rtol=1e-6)
        assert float(reg(scaled, {})) == pytest.approx(expected, rel=1e-6)
    zeros = {
        "sample": jnp.zeros((3, 4), dtype=jnp.complex64),
        "probe_modes": jnp.zeros((2, 3, 4), dtype=jnp.complex64),
    }
    assert all(
        np.isfinite(np.asarray(x)).all()
        for x in jax.tree.leaves(jax.grad(reg)(zeros, {}))
    )


def test_cosine_support_preserves_interior_phase_and_zeros_exterior_without_mutation():
    mask = projectors.create_smooth_spatial_support(
        (11, 11), radius_px=2.0, decay_px=2.0
    )
    # At radii 0, 2, 3, 4 the support is 1, 1, 1/2, 0.
    np.testing.assert_allclose(
        np.asarray(mask)[5, [5, 7, 8, 9]], [1, 1, 0.5, 0], atol=1e-6
    )
    params = {"probe_modes": jnp.full((5, 11, 11), 2 + 3j), "sample": jnp.ones((3, 3))}
    fixed = {"positions": jnp.array([1, 2])}
    projection = projectors.create_spatial_support_projection(mask)
    projected, fixed_out = projection(params, fixed)
    np.testing.assert_array_equal(params["probe_modes"], 2 + 3j)
    np.testing.assert_array_equal(projected["probe_modes"][:, 5, 5], 2 + 3j)
    np.testing.assert_allclose(projected["probe_modes"][:, 5, 8], 1 + 1.5j, atol=1e-6)
    np.testing.assert_array_equal(projected["probe_modes"][:, 5, 9], 0)
    np.testing.assert_array_equal(projected["sample"], params["sample"])
    np.testing.assert_array_equal(fixed_out["positions"], fixed["positions"])


def test_support_is_applied_after_each_optimizer_update():
    mask = projectors.create_smooth_spatial_support((7, 7), 1.0, 1.0)
    projection = projectors.create_spatial_support_projection(mask)
    params = {"probe_modes": jnp.zeros((2, 7, 7))}
    loss = jax.jit(
        jax.value_and_grad(lambda p, f, i, m, mask: -jnp.sum(p["probe_modes"]))
    )
    optimizer = optax.sgd(1.0)
    final, _, _, _ = run_optimization_loop(
        params,
        optimizer.init(params),
        optimizer,
        loss,
        {},
        np.ones((3, 7, 7), dtype=np.float32),
        jnp.ones((7, 7), dtype=bool),
        "accumulate_full_pass",
        3,
        2,
        projection_fn=projection,
    )
    assert float(final["probe_modes"][0, 3, 3]) == pytest.approx(3.0)
    np.testing.assert_array_equal(final["probe_modes"][:, 0, 0], 0)


def test_full_pipeline_saves_named_artifacts_objective_and_support(tmp_path):
    pipeline = importlib.import_module("pipeline_nanomax_upd_v2")
    path = tmp_path / "scan.h5"
    y, x = np.indices((24, 24))
    image = (500 * np.exp(-((y - 12) ** 2 + (x - 12) ** 2) / 14) + 1).astype(np.uint32)
    frames = np.stack([image, 2 * image, 3 * image, 4 * image])
    frames[:, 0, 0] = np.iinfo(np.uint32).max
    with h5py.File(path, "w") as f:
        f["/entry/measurement/eiger4m/frames"] = frames
        f["/entry/measurement/sx"] = [-0.1, 0.1, -0.1, 0.1]
        f["/entry/measurement/sy"] = [-0.1, -0.1, 0.1, 0.1]
        f["/entry/snapshots/pre_scan/energy_raw"] = [10000.0]
    config = pipeline.ReconstructionConfig(
        datapath=path,
        output_dir=tmp_path / "results",
        tag="both",
        crop_shape=(16, 16),
        crop_center=(12, 12),
        n_steps=4,
        batch_size=3,
        shift_margin=3,
        max_correction_magnitude=2.0,
        object_padding=4,
        sample_tv_weight=1.0,
        probe_tv_weight=0.01,
        probe_support=True,
        probe_support_radius_px=4.0,
        probe_support_decay_px=2.0,
        regularization_schedule=("1:4:2",),
        projection_schedule=("2:4",),
        z_samples=9,
        padding_factor=2,
    )
    metrics = pipeline.run_pipeline(config, [2])
    assert metrics[0]["epochs"] == 4
    assert metrics[0]["final_objective"] == pytest.approx(
        metrics[0]["final_loss"] + metrics[0]["final_regularization"]
    )
    assert metrics[0]["final_regularization"] > 0
    assert metrics[0]["regularization_active_steps"] == 2
    assert metrics[0]["projection_active_steps"] == 2
    assert np.isfinite(metrics[0]["diagnostics"]["estimated_focus_z_m"])
    files = list(config.run_dir.rglob("*"))
    assert all(p.name.startswith(config.run_id + "_") for p in files if p.is_file())
    assert config.artifact("results.md").exists()
    report = config.artifact("results.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)", report):
        assert (config.run_dir / target).is_file(), target
    folder = config.run_dir / "modes_2"
    with np.load(config.artifact("reconstruction.npz", folder)) as saved:
        assert len(saved["loss_history"]) == 4
        np.testing.assert_array_equal(
            saved["regularization_active_history"], [False, True, False, True]
        )
        np.testing.assert_array_equal(
            saved["projection_active_history"], [False, False, True, True]
        )
        assert int(saved["optimization_steps_completed"]) == 4
        np.testing.assert_array_equal(saved["objective_history"], saved["loss_history"])
        assert np.isfinite(saved["sample"]).all()
        assert saved["probe_support"].shape == (16, 16)
        assert np.max(np.abs(saved["probe_modes"][:, saved["probe_support"] == 0])) == 0
    saved_config = json.loads(config.artifact("configuration.json").read_text())
    assert saved_config["tag"] == "both"
    assert saved_config["sample_tv_weight"] == 1.0
    assert saved_config["source_sha256"]
    diagnostic_folder = folder / "diagnostics"
    report = config.artifact("results.md", diagnostic_folder).read_text()
    for target in re.findall(r"\]\(([^)]+)\)", report):
        if not target.startswith("https://"):
            assert (diagnostic_folder / target).is_file()
    with np.load(config.artifact("diagnostics.npz", diagnostic_folder)) as analysis:
        with np.load(config.artifact("reconstruction.npz", folder)) as original:
            fields = np.einsum(
                "kyx,jkm->jmyx", original["probe_modes"], original["modal_weights"]
            )
            rotated = np.einsum(
                "kyx,jkm->jmyx",
                analysis["basis_coherent_modes"],
                analysis["basis_coherent_weights"],
            )
            np.testing.assert_allclose(rotated, fields, rtol=1e-5, atol=1e-4)
            np.testing.assert_array_equal(
                analysis["scan_frame_indices"], np.sort(original["frame_indices"])
            )
        assert analysis["propagation_z_m"].shape == (9,)
    for name, expected_hash in saved_config["source_sha256"].items():
        relative = Path(name)
        snapshot = config.artifact(
            relative.name, config.run_dir / "source_snapshot" / relative.parent
        )
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected_hash
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(config, [2])
