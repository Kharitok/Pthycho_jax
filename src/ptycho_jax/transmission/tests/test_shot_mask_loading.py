"""Per-shot high-value masks must survive NanoMAX preparation."""

import h5py
import numpy as np
import pytest

from ptycho_jax.data_loading import nanomax_dataloader as nm


@pytest.fixture
def make_dataset(tmp_path):
    def make(frames, coords=None):
        frames = np.asarray(frames)
        if coords is None:
            coords = np.column_stack((np.arange(len(frames)), np.zeros(len(frames))))
        path = tmp_path / "scan.h5"
        with h5py.File(path, "w") as f:
            f[nm.NanoMaxDataset.detector_path] = frames
            f[nm.NanoMaxDataset.scan_x_path] = np.asarray(coords)[:, 0]
            f[nm.NanoMaxDataset.scan_y_path] = np.asarray(coords)[:, 1]
            f[nm.NanoMaxDataset.energy_path] = 10000.0
        return nm.NanoMaxDataset(path, 2.0, 1e-4, -1e-3)

    return make


@pytest.mark.parametrize("fft_shift", [False, True])
def test_shot_masks_follow_moving_maxima_filters_and_repeat_order(
    make_dataset, fft_shift
):
    frames = np.broadcast_to(np.arange(10, 60, 10)[:, None, None], (5, 6, 6)).copy()
    for i, pixel in enumerate([(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]):
        frames[i, pixel[0], pixel[1]] = 999
    dataset = make_dataset(frames, [[1, 0], [0, 0], [1, 0], [0, 0], [9, 9]])
    precise = np.ones((6, 6), bool)
    precise[4, 4] = False
    result = nm.prepare_nanomax_data(
        dataset,
        mask=np.ones((6, 6), bool),
        precise_mask=precise,
        shot_specific_masks=True,
        crop_shape=(4, 4),
        crop_center=(3, 3),
        scan_coord_limits={"0": (0, 1)},
        intensity_limits=(270, 600),
        fft_shift=fft_shift,
    )
    np.testing.assert_array_equal(result["frame_indices"], [3, 2, 1])
    np.testing.assert_array_equal(result["total_ints"], [560, 420, 280])
    expected = np.ones((3, 4, 4), bool)
    expected[:, 3, 3] = False
    expected[0, 1, 1] = expected[1, 1, 0] = expected[2, 0, 1] = False
    mean = np.array(
        [[35, 35, 35, 35], [40, 30, 35, 35], [35, 35, 35, 35], [35, 35, 35, 0]]
    )
    if fft_shift:
        expected = np.fft.ifftshift(expected, axes=(-2, -1))
        mean = np.fft.ifftshift(mean)
    np.testing.assert_array_equal(result["shot_masks"], expected)
    np.testing.assert_array_equal(result["detector_mask"], expected.any(axis=0))
    assert np.max(result["measured_intensities"]) == 40
    assert np.all(result["measured_intensities"][~expected] == 0)
    np.testing.assert_array_equal(result["mean_image"], mean)


def test_shot_heuristic_rejects_own_tied_maxima_and_nonfinite_values(make_dataset):
    frames = np.array([[[999, 2, -2], [4, np.nan, 6]], [[1, 999, 3], [999, np.inf, 7]]])
    result = nm.prepare_nanomax_data(make_dataset(frames), shot_specific_masks=True)
    np.testing.assert_array_equal(
        result["shot_masks"],
        [
            [[False, True, False], [True, False, True]],
            [[True, False, True], [False, False, True]],
        ],
    )
    np.testing.assert_array_equal(result["total_ints"], [12, 11])
    # Frame zero's invalid pixels may be usable in a different shot.
    assert result["shot_masks"][1, 0, 0]


def test_empty_shot_mask_is_rejected_before_loss_can_be_nan(make_dataset):
    frames = np.arange(1, 33).reshape(2, 4, 4)
    frames[1] = 999
    with pytest.raises(ValueError, match="[Ff]rame.*1.*valid|valid.*[Ff]rame.*1"):
        nm.prepare_nanomax_data(make_dataset(frames), shot_specific_masks=True)
