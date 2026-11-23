"""JSON inputs must preserve CLI precedence, strict types and run provenance."""

import hashlib
import json
import re

import h5py
import numpy as np
import pytest

import pipeline_nanomax_upd_v2 as pipeline


def test_json_settings_load_defaults_paths_lists_and_cli_overrides(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "settings"
    config_dir.mkdir()
    working = tmp_path / "working"
    working.mkdir()
    path = config_dir / "scan.json"
    path.write_text(
        json.dumps(
            {
                "datapath": "raw.h5",
                "output_dir": "runs",
                "n_steps": 20,
                "modes": [5],
                "device": "cpu",
                "tag": "from_json",
                "probe_support": True,
                "sample_tv_weight": 1.2,
                "regularization_schedule": ["2:10:2", "15"],
                "projection_schedule": ["5:20:3", "19"],
                "crop_shape": [32, 32],
                "diagnostics": False,
                "scan_coord_limits": {"0": [-1, 2]},
                "intensity_limits": [0, 1000000],
            }
        )
    )
    monkeypatch.chdir(working)
    config, modes, device, source = pipeline.parse_configuration(
        [
            "--config",
            str(path),
            "--steps",
            "25",
            "--sample-tv-weight",
            "0",
            "--no-probe-support",
            "--output-dir",
            "override",
            "--modes",
            "1",
            "2",
        ]
    )
    assert config.datapath == config_dir / "raw.h5"
    assert config.output_dir.resolve() == working / "override"
    assert config.n_steps == 25 and config.sample_tv_weight == 0
    assert not config.probe_support and not config.diagnostics
    assert config.crop_shape == (32, 32)
    assert config.regularization_schedule == ("2:10:2", "15")
    assert config.intensity_limits == (0, 1000000)
    assert config.scan_coord_limits == {"0": (-1, 2)}
    assert list(modes) == [1, 2] and device == "cpu"
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source["content"] == path.read_bytes()
    assert config.seed == 0


def test_json_null_optional_values_and_steps_alias(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "steps": 8,
                "tag": None,
                "device": None,
                "crop_center": None,
                "regularization_schedule": None,
                "projection_schedule": ["none"],
            }
        )
    )
    config, modes, device, _ = pipeline.parse_configuration(["--config", str(path)])
    assert config.n_steps == 8
    assert re.fullmatch(r"[A-Za-z]{5}", config.tag)
    assert config.crop_center is None and config.regularization_schedule is None
    assert config.projection_schedule == ("none",)
    assert list(modes) == [1, 5] and device is None


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "{",
        '{"n_steps": 5, "n_steps": 6}',
        '{"sample_tv_weight": NaN}',
        '{"sample_tv_weight": 1e999}',
        '{"sample_tv_weight": "1.0"}',
        '{"diagnostics": "false"}',
        '{"probe_support": 1}',
        '{"n_steps": true}',
        '{"crop_shape": [16]}',
        '{"crop_shape": [16, 16.0]}',
        '{"regularization_schedule": "10:20"}',
        '{"regularization_schedule": []}',
        '{"sample_tv_weigth": 1}',
        '{"started_utc": "2020-01-01T00:00:00Z"}',
        '{"n_steps": 5, "steps": 5}',
        '{"modes": [true]}',
        '{"modes": [0]}',
        '{"modes": [2, 2]}',
        '{"device": "tpu"}',
        '{"datapath": null}',
        '{"scan_coord_limits": {"x": [0, 1]}}',
        '{"scan_coord_limits": {"0": [2, 1]}}',
        '{"intensity_limits": [10, 1]}',
        '{"probe_support_radius_px": -2}',
        '{"n_steps": 5, "regularization_schedule": ["0:10"]}',
        '{"sample_learning_rate": -1}',
        '{"relative_probe_threshold": 1}',
        '{"detector_distance_m": 0}',
        '{"crop_shape": [0, 16]}',
    ],
)
def test_bad_json_is_rejected_before_creating_outputs(tmp_path, text, capsys):
    path = tmp_path / "bad.json"
    path.write_text(text)
    output = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit) as error:
        pipeline.main(["--config", str(path), "--output-dir", str(output)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "error:" in message
    assert "unrecognized arguments: --config" not in message
    assert not output.exists()


def test_json_only_cli_runs_reconstruction_and_preserves_input(tmp_path):
    y, x = np.indices((24, 24))
    image = (500 * np.exp(-((y - 12) ** 2 + (x - 12) ** 2) / 14) + 1).astype(np.uint32)
    frames = np.stack([image, 2 * image, 3 * image])
    frames[:, 0, 0] = np.iinfo(np.uint32).max
    with h5py.File(tmp_path / "raw.h5", "w") as f:
        f["/entry/measurement/eiger4m/frames"] = frames
        f["/entry/measurement/sx"] = [-0.1, 0, 0.1]
        f["/entry/measurement/sy"] = [-0.1, 0.1, 0]
        f["/entry/snapshots/pre_scan/energy_raw"] = [10000.0]
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "datapath": "raw.h5",
                "output_dir": "runs",
                "tag": "json_test",
                "device": "cpu",
                "modes": [2],
                "n_steps": 3,
                "batch_size": 2,
                "crop_shape": [16, 16],
                "crop_center": [12, 12],
                "shift_margin": 3,
                "max_correction_magnitude": 2.0,
                "object_padding": 4,
                "sample_tv_weight": 1.0,
                "regularization_schedule": ["1:3"],
                "probe_support": True,
                "probe_support_radius_px": 4.0,
                "probe_support_decay_px": 2.0,
                "projection_schedule": ["2"],
                "diagnostics": True,
                "z_samples": 5,
                "padding_factor": 1,
            },
            indent=2,
        )
        + "\n"
    )
    assert pipeline.main(["--config", str(path)]) == 0
    (folder,) = (tmp_path / "runs").iterdir()
    saved_config = json.loads(next(folder.glob("*_configuration.json")).read_text())
    source = saved_config["settings_source"]
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source["path"] == str(path)
    assert (folder / source["snapshot"]).read_bytes() == path.read_bytes()
    assert saved_config["n_steps"] == 3 and saved_config["modes"] == [2]
    assert saved_config["diagnostics"] is True
    metrics = json.loads(next((folder / "modes_2").glob("*_metrics.json")).read_text())
    assert metrics["regularization_active_steps"] == 2
    assert metrics["projection_active_steps"] == 1
    assert np.isfinite(metrics["diagnostics"]["estimated_focus_z_m"])
    assert all(
        p.name.startswith(saved_config["run_id"] + "_")
        for p in folder.rglob("*")
        if p.is_file()
    )
