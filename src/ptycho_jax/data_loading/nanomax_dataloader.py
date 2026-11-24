"""Read NanoMAX HDF5 scans and prepare aligned inputs for reconstruction.

All detector masks use True for valid pixels. Detector crops use (row, column)
coordinates; scan limits use the original (sx, sy) motor coordinates and units.
See nanomax_data_loading.md for the processing order and pipeline integration.
"""

import re
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import c, electron_volt, h
from scipy.ndimage import center_of_mass

try:
    import hdf5plugin  # noqa: F401 -- registers compressed HDF5 frame readers
except ModuleNotFoundError:
    hdf5plugin = None


class H5DataLoader:
    """Context-managed HDF5 access, including scalar metadata and sliced reads."""

    def __init__(self, filepath: str | Path, mode: str = "r") -> None:
        self.filepath = Path(filepath)
        self.mode = mode
        self._file: h5py.File | None = None

    def __enter__(self) -> "H5DataLoader":
        if self._file is not None:
            raise RuntimeError("H5DataLoader context is already open.")
        self._file = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("Cannot access 'self.file' outside of a 'with' block.")
        return self._file

    def read_dataset(self, key: str, selection=()):
        try:
            return self.file[key][selection]
        except OSError as exc:
            if hdf5plugin is None and any(s in str(exc).lower() for s in ("filter", "plugin")):
                raise OSError(
                    f"Cannot read compressed dataset {key!r}; install hdf5plugin "
                    "in this Python environment (uv pip install hdf5plugin)."
                ) from exc
            raise


class NanoMaxDataset:
    """Raw NanoMAX data with externally supplied geometry in metres.

    detector_name selects /entry/measurement/<name>/frames. HDF5 external links
    are followed automatically, keeping motors/energy in the scan file. Omit
    the name to retain the class path (eiger4m by default) or subclass layout.
    Raw properties return in-memory arrays; detector_mask reads only frame zero.
    """

    detector_path = "/entry/measurement/eiger4m/frames"
    scan_x_path = "/entry/measurement/sx"
    scan_y_path = "/entry/measurement/sy"
    energy_path = "/entry/snapshots/pre_scan/energy_raw"

    def __init__(
        self,
        datapath: str | Path,
        detector_distance: float,
        detector_pixel_size: float,
        defocus: float,
        *,
        detector_name: str | None = None,
    ) -> None:
        self.datapath = Path(datapath)
        if detector_name is not None:
            self.detector_path = self.detector_path_for_name(detector_name)
        path_parts = self.detector_path.strip("/").split("/")
        self.detector_name = detector_name or (path_parts[-2] if len(path_parts) > 1 else None)
        self.detector_distance = _positive_scalar(detector_distance, "detector_distance")
        self.detector_pixel_size = _positive_scalar(detector_pixel_size, "detector_pixel_size")
        if not np.isscalar(defocus) or not np.isfinite(defocus):
            raise ValueError("defocus must be a finite scalar in metres.")
        self.defocus = float(defocus)

    @staticmethod
    def detector_path_for_name(name: str) -> str:
        """Validate a detector key, without accepting arbitrary HDF5 paths."""
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("detector_name must contain letters, digits, underscores or hyphens.")
        return f"/entry/measurement/{name}/frames"

    def __getitem__(self, key: str):
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(key)

    @property
    def detector_data(self) -> np.ndarray:
        return self[self.detector_path]

    @property
    def scan_x_data(self) -> np.ndarray:
        return self[self.scan_x_path]

    @property
    def scan_y_data(self) -> np.ndarray:
        return self[self.scan_y_path]

    @property
    def energy_data(self):
        return self[self.energy_path]

    @property
    def detector_mask(self) -> np.ndarray:
        with H5DataLoader(self.datapath) as loader:
            return _first_frame_mask(loader.read_dataset(self.detector_path, 0))


def _positive_scalar(value, name: str) -> float:
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return float(value)


def _first_frame_mask(frame: np.ndarray) -> np.ndarray:
    finite = np.isfinite(frame)
    if not finite.any():
        return np.zeros(frame.shape, dtype=bool)
    return finite & (frame >= 0) & (frame < frame[finite].max())


def _detector_image(value, shape, name: str, *, boolean=False) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != shape or not np.isfinite(image).all():
        raise ValueError(f"{name} must be finite and have full detector shape {shape}.")
    return image.astype(bool) if boolean else image


def _clean_image(frame: np.ndarray, mask: np.ndarray, dtype) -> np.ndarray:
    with np.errstate(over="ignore"):
        clean = np.array(frame, dtype=dtype, copy=True)
    np.nan_to_num(clean, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(clean, 0, out=clean)
    clean[~mask] = 0
    return clean


def mean_valid_intensity(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Mean at each pixel over only the shots that observed that pixel."""
    counts = np.broadcast_to(masks, images.shape).sum(axis=0)
    total = np.where(masks, images, 0).sum(axis=0, dtype=np.float64)
    return np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)


def _within_limits(values: np.ndarray, limits, name: str) -> np.ndarray:
    bounds = np.asarray(limits, dtype=float)
    if bounds.shape != (2,) or np.isnan(bounds).any() or bounds[0] > bounds[1]:
        raise ValueError(f"{name} must be an ordered (lower, upper) pair.")
    return (values >= bounds[0]) & (values <= bounds[1])


def _estimate_center(image: np.ndarray, radius: int, threshold: float) -> np.ndarray:
    if image.max() <= 0:
        raise ValueError("Cannot estimate a detector crop center from an empty mean image.")
    center = np.asarray(center_of_mass(image))
    origin = np.maximum(np.floor(center).astype(int) - radius, 0)
    stop = np.minimum(np.floor(center).astype(int) + radius, image.shape)
    patch = image[tuple(slice(int(a), int(b)) for a, b in zip(origin, stop))]
    support = patch > patch.max() * threshold
    if not support.any():
        raise ValueError("Cannot refine detector crop center from an empty patch.")
    return origin + np.asarray(center_of_mass(support))


def _crop_slices(shape, crop_shape, center) -> tuple[slice, slice]:
    start = np.floor(center).astype(int) - np.asarray(crop_shape) // 2
    stop = start + crop_shape
    if np.any(start < 0) or np.any(stop > shape):
        raise ValueError("Requested detector crop does not fit; change crop_shape or crop_center.")
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop))


def _repeat_order(position_id: np.ndarray, totals: np.ndarray) -> np.ndarray:
    # Rank brightness within each coordinate group, then interleave groups by rank.
    grouped = np.lexsort((-totals, position_id))
    positions = position_id[grouped]
    starts = np.flatnonzero(np.r_[True, positions[1:] != positions[:-1]])
    sizes = np.diff(np.r_[starts, len(grouped)])
    rank = np.arange(len(grouped)) - np.repeat(starts, sizes)
    return grouped[np.lexsort((positions, rank))]


def prepare_nanomax_data(
    dataset: NanoMaxDataset,
    *,
    scan_coord_limits: Mapping[str, tuple[float, float]] | None = None,
    intensity_limits: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    precise_mask: np.ndarray | None = None,
    shot_specific_masks: bool = False,
    probe_image: np.ndarray | None = None,
    crop_shape: tuple[int, int] | None = None,
    crop_center: tuple[float, float] | None = None,
    center_refinement_radius: int = 200,
    center_relative_threshold: float = 0.1,
    scan_coords_unit: float = 1e-6,
    energy_unit: float = 1.0,
    scan_axis_order: tuple[int, int] = (0, 1),
    fft_shift: bool = False,
    dtype=np.float32,
) -> dict:
    """Prepare frames, detector masks, metadata and scan offsets without JAX.

    scan_coord_limits uses keys '0'/'1' for original sx/sy and inclusive limits
    in motor units. intensity_limits applies to cleaned, masked, cropped totals.
    With shot_specific_masks=False, mask replaces the first-frame maximum
    heuristic; precise_mask intersects it. With True, each full raw frame gets
    the same finite/nonnegative/below-frame-maximum heuristic, intersected with
    any supplied static masks. shot_masks follows the retained frame order;
    detector_mask is its 2D union for display and shared probe initialization.
    Masks and probe_image must have full detector shape and detector orientation.
    dtype must be float32 or a wider floating-point type to retain photon counts.

    crop_center is (row, column) in the original detector. If omitted with a
    crop_shape, estimate and refine it from the coordinate-selected mean image.
    Frames are read individually, keeping only the cropped stack in memory.

    scan_coords_unit converts motor values to metres; energy_unit converts energy
    to eV. scan_axis_order maps motor columns onto array axes for pixel offsets.
    Raw scan_coords remain (sx, sy). Rectangular crops use two sample pixel sizes.

    Returns the original pipeline's scan_coords, total_ints,
    measured_intensities, position_id, mean_image, mean_probe, mask and
    precise_mask fields, plus detector_mask, frame_indices, crop_slices,
    detector_center, experimental_parameters, reconstruction_pix_size,
    int_parts, float_parts, min_required_sample_size_pix and sample_size_pix.
    Arrays are centered unless fft_shift=True; mean_probe always stays centered.
    """
    scan_coords_unit = _positive_scalar(scan_coords_unit, "scan_coords_unit")
    if type(shot_specific_masks) is not bool:
        raise ValueError("shot_specific_masks must be a boolean.")
    energy_unit = _positive_scalar(energy_unit, "energy_unit")
    if tuple(scan_axis_order) not in ((0, 1), (1, 0)):
        raise ValueError("scan_axis_order must be (0, 1) or (1, 0).")
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating) or dtype.itemsize < 4:
        raise ValueError("dtype must be float32 or a wider floating-point type.")
    if (not isinstance(center_refinement_radius, (int, np.integer))
            or center_refinement_radius < 1):
        raise ValueError("center_refinement_radius must be a positive integer.")
    if not np.isfinite(center_relative_threshold) or not 0 <= center_relative_threshold < 1:
        raise ValueError("center_relative_threshold must be in [0, 1).")
    if crop_center is not None and crop_shape is None:
        raise ValueError("crop_center requires crop_shape.")

    with H5DataLoader(dataset.datapath) as loader:
        frames = loader.file[dataset.detector_path]
        if frames.ndim != 3 or any(size == 0 for size in frames.shape):
            raise ValueError("Detector frames must have nonempty shape (frames, rows, columns).")
        detector_shape = frames.shape[-2:]
        detector_source = {
            "name": dataset.detector_name,
            "scan_file": str(dataset.datapath.resolve()),
            "scan_dataset": dataset.detector_path,
            "data_file": str(Path(frames.file.filename).resolve()),
            "data_dataset": frames.name,
            "shape": list(frames.shape),
        }
        x = loader.read_dataset(dataset.scan_x_path)
        y = loader.read_dataset(dataset.scan_y_path)
        if x.shape != (len(frames),) or y.shape != (len(frames),):
            raise ValueError("Each scan coordinate array must contain one value per detector frame.")
        coords = np.column_stack((x, y))
        energy = np.asarray(loader.read_dataset(dataset.energy_path))
        if energy.size != 1:
            raise ValueError("energy must be scalar or contain exactly one value.")
        energy_eV = _positive_scalar(energy.item() * energy_unit, "energy")
        wavelength = h * c / (energy_eV * electron_volt)

        valid_coords = np.isfinite(coords).all(axis=1)
        if scan_coord_limits is not None:
            if set(scan_coord_limits) - {"0", "1"}:
                raise ValueError("scan_coord_limits keys must be '0' and/or '1'.")
            for axis, limits in scan_coord_limits.items():
                valid_coords &= _within_limits(coords[:, int(axis)], limits, "scan_coord_limits")
        frame_indices = np.flatnonzero(valid_coords)
        if not frame_indices.size:
            raise ValueError("No frames remain after scan coordinate filtering.")

        if mask is not None:
            full_mask = _detector_image(mask, detector_shape, "mask", boolean=True)
        elif shot_specific_masks:
            full_mask = np.ones(detector_shape, dtype=bool)
        else:
            full_mask = _first_frame_mask(loader.read_dataset(dataset.detector_path, 0))
        full_precise = None if precise_mask is None else _detector_image(
            precise_mask, detector_shape, "precise_mask", boolean=True,
        )
        combined_mask = full_mask.copy()
        if full_precise is not None:
            combined_mask &= full_precise
        if not combined_mask.any():
            raise ValueError("The detector mask has no valid pixels; supply an explicit mask.")
        full_probe = None if probe_image is None else _detector_image(
            probe_image, detector_shape, "probe_image",
        )

        center = None
        slices = (slice(0, detector_shape[0]), slice(0, detector_shape[1]))
        if crop_shape is not None:
            crop_size = np.asarray(crop_shape)
            if (crop_size.shape != (2,) or not np.issubdtype(crop_size.dtype, np.integer)
                    or np.any(crop_size <= 0) or np.any(crop_size > detector_shape)):
                raise ValueError("crop_shape must be two positive integers within detector shape.")
            if crop_center is None:
                mean = np.zeros(detector_shape, dtype=np.float64)
                observations = np.zeros(detector_shape, dtype=np.int64) if shot_specific_masks else None
                for index in frame_indices:
                    raw = loader.read_dataset(dataset.detector_path, int(index))
                    valid = combined_mask & _first_frame_mask(raw) if shot_specific_masks else combined_mask
                    mean += _clean_image(raw, valid, dtype)
                    if observations is not None:
                        observations += valid
                if observations is None:
                    mean /= len(frame_indices)
                else:
                    np.divide(mean, observations, out=mean, where=observations > 0)
                center = _estimate_center(mean, center_refinement_radius, center_relative_threshold)
            else:
                center = np.asarray(crop_center, dtype=float)
                if center.shape != (2,) or not np.isfinite(center).all():
                    raise ValueError("crop_center must be a finite (row, column) pair.")
            slices = _crop_slices(detector_shape, crop_size, center)

        cropped_mask = combined_mask[slices]
        if not cropped_mask.any():
            raise ValueError("The detector crop contains no valid mask pixels.")
        measured = np.empty((len(frame_indices), *cropped_mask.shape), dtype=dtype)
        shot_masks = np.empty(measured.shape, dtype=bool) if shot_specific_masks else None
        for output_index, frame_index in enumerate(frame_indices):
            if shot_masks is None:
                raw = loader.read_dataset(dataset.detector_path, (int(frame_index), *slices))
                valid = cropped_mask
            else:
                full_raw = loader.read_dataset(dataset.detector_path, int(frame_index))
                valid = _first_frame_mask(full_raw)[slices] & cropped_mask
                if not valid.any():
                    raise ValueError(f"Frame {frame_index} has no valid pixels in the detector crop.")
                shot_masks[output_index] = valid
                raw = full_raw[slices]
            measured[output_index] = _clean_image(raw, valid, dtype)

    totals = measured.sum(axis=(-2, -1), dtype=np.float64)
    if intensity_limits is not None:
        valid_intensity = _within_limits(totals, intensity_limits, "intensity_limits")
        frame_indices = frame_indices[valid_intensity]
        measured = measured[valid_intensity]
        totals = totals[valid_intensity]
        if shot_masks is not None:
            shot_masks = shot_masks[valid_intensity]
    if not len(frame_indices):
        raise ValueError("No frames remain after intensity filtering.")
    coords = coords[frame_indices]
    unique, position_id = np.unique(coords, axis=0, return_inverse=True)
    order = _repeat_order(position_id, totals)
    measured = measured[order]
    if shot_masks is not None:
        shot_masks = shot_masks[order]
        cropped_mask = shot_masks.any(axis=0)
        mean_image = mean_valid_intensity(measured[:len(unique)], shot_masks[:len(unique)])
    else:
        mean_image = measured[:len(unique)].mean(axis=0)
    mean_probe = mean_image.copy() if full_probe is None else _clean_image(
        full_probe[slices], cropped_mask, dtype,
    )

    pixel_number = np.asarray(cropped_mask.shape)
    pixel_size = wavelength * dataset.detector_distance / (dataset.detector_pixel_size * pixel_number)
    coords_pix = coords[order][:, scan_axis_order] * scan_coords_unit / pixel_size
    coords_pix -= coords_pix.mean(axis=0)
    int_parts = np.floor(coords_pix).astype(np.int64)
    float_parts = coords_pix - int_parts
    minimum_size = np.ptp(int_parts, axis=0) + 1 + pixel_number
    result = {
        "scan_coords": coords[order],
        "total_ints": totals[order],
        "measured_intensities": measured,
        "position_id": position_id[order],
        "mean_image": mean_image,
        "mean_probe": mean_probe,
        "mask": full_mask[slices],
        "precise_mask": None if full_precise is None else full_precise[slices],
        "detector_mask": cropped_mask,
        "shot_masks": shot_masks,
        "frame_indices": frame_indices[order],
        "crop_slices": slices,
        "detector_center": center,
        "detector_source": detector_source,
        "experimental_parameters": {
            "detector_pixel_size_m": dataset.detector_pixel_size,
            "pixel_number": pixel_number,
            "sample_detector_distance_m": dataset.detector_distance,
            "wavelength_m": wavelength,
            "energy_eV": energy_eV,
            "assumed_defocus_at_sample_m": dataset.defocus,
        },
        "reconstruction_pix_size": pixel_size,
        "int_parts": int_parts,
        "float_parts": float_parts,
        "min_required_sample_size_pix": minimum_size,
        "sample_size_pix": (minimum_size * 1.2).astype(np.int64),
    }
    if fft_shift:
        for key in ("measured_intensities", "mean_image", "mask", "precise_mask", "detector_mask", "shot_masks"):
            if result[key] is not None:
                result[key] = np.fft.ifftshift(result[key], axes=(-2, -1))
    return result
