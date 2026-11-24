"""Detector names select linked NanoMAX frame files without mixing metadata."""

import json

import h5py
import numpy as np
import pytest

import pipeline_nanomax_upd_v2 as pipeline
from ptycho_jax.data_loading.nanomax_dataloader import (
    NanoMaxDataset,
    prepare_nanomax_data,
)


@pytest.fixture
def linked_scan(tmp_path):
    detector = tmp_path / "scan_000005_eiger1m.h5"
    scan = tmp_path / "000005.h5"
    y, x = np.indices((24, 24))
    image = (500 * np.exp(-((y - 12) ** 2 + (x - 12) ** 2) / 14) + 1).astype(np.uint32)
    frames = np.stack([image, 2 * image, 3 * image])
    frames[:, 0, 0] = np.iinfo(np.uint32).max
    with h5py.File(detector, "w") as f:
        f["entry/instrument/Eiger/data"] = frames
    with h5py.File(scan, "w") as f:
        f["entry/measurement/eiger1m/frames"] = h5py.ExternalLink(
            detector.name, "/entry/instrument/Eiger/data"
        )
        f["entry/measurement/sx"] = [-0.1, 0.1, 0.0]
        f["entry/measurement/sy"] = [-0.1, 0.1, 0.0]
        f["entry/snapshots/pre_scan/energy_raw"] = [15949.18569841]
    return scan, detector, frames


def test_eiger1m_external_link_reads_matching_frames_and_scan_metadata(linked_scan):
    scan, detector, frames = linked_scan
    dataset = NanoMaxDataset(scan, 3.635, 75e-6, 0.001, detector_name="eiger1m")
    assert dataset.detector_path == "/entry/measurement/eiger1m/frames"
    np.testing.assert_array_equal(dataset.detector_data, frames)
    assert not dataset.detector_mask[0, 0]
    data = prepare_nanomax_data(dataset, crop_shape=(16, 16), crop_center=(12, 12))
    np.testing.assert_array_equal(
        data["measured_intensities"], frames[data["frame_indices"], 4:20, 4:20]
    )
    np.testing.assert_array_equal(
        data["scan_coords"],
        np.array([[-0.1, -0.1], [0.1, 0.1], [0.0, 0.0]])[data["frame_indices"]],
    )
    assert data["experimental_parameters"]["energy_eV"] == pytest.approx(15949.18569841)
    source = data["detector_source"]
    assert source["name"] == "eiger1m"
    assert source["data_file"] == str(detector)
    assert source["data_dataset"] == "/entry/instrument/Eiger/data"
    assert source["scan_dataset"] == "/entry/measurement/eiger1m/frames"


@pytest.mark.parametrize("dataset_path", ["/custom/data", "/frames"])
def test_default_and_subclass_detector_layouts_remain_supported(tmp_path, dataset_path):
    assert NanoMaxDataset.detector_path == "/entry/measurement/eiger4m/frames"
    with h5py.File(tmp_path / "test.h5", "w") as f:
        f[dataset_path] = np.ones((1, 2, 2))

    class CustomDataset(NanoMaxDataset):
        detector_path = dataset_path

    dataset = CustomDataset(tmp_path / "test.h5", 1.0, 75e-6, 0.001)
    np.testing.assert_array_equal(dataset.detector_data, np.ones((1, 2, 2)))


@pytest.mark.parametrize("name", ["", "../eiger1m", "eiger1m/frames", "a b"])
def test_invalid_detector_names_rejected_before_output(name):
    with pytest.raises(ValueError):
        pipeline.ReconstructionConfig(detector_name=name)


def test_json_cli_detector_selection_runs_real_ad_and_diagnostics(
    linked_scan, tmp_path
):
    scan, detector, _ = linked_scan
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "datapath": scan.name,
                "output_dir": "runs",
                "detector_name": "eiger1m",
                "device": "cpu",
                "modes": [2],
                "n_steps": 3,
                "batch_size": 2,
                "tag": "linked_detector",
                "detector_distance_m": 3.635,
                "defocus_m": 0.001,
                "crop_shape": [16, 16],
                "crop_center": [12, 12],
                "object_padding": 4,
                "shift_margin": 3,
                "max_correction_magnitude": 2.0,
                "z_samples": 5,
                "padding_factor": 1,
            }
        )
    )
    parsed, _, _, _ = pipeline.parse_configuration(
        ["--config", str(path), "--detector-name", "eiger1m"]
    )
    assert parsed.detector_name == "eiger1m"
    assert pipeline.main(["--config", str(path)]) == 0
    (folder,) = (tmp_path / "runs").iterdir()
    metrics = json.loads(next((folder / "modes_2").glob("*_metrics.json")).read_text())
    assert metrics["detector_source"]["data_file"] == str(detector)
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert np.isfinite(metrics["diagnostics"]["estimated_focus_z_m"])
