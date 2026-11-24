"""Reconstruct a NanoMAX scan with the existing JAX transmission model.

Example: .venv/bin/python src/pipeline_nanomax_upd.py --modes 1 5 --steps 400
Figures, arrays, configuration and measured fit statistics are saved per run.
The original pipeline scripts and the shared reconstruction modules are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Share the workstation without reserving most of a GPU at startup.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptycho_jax.data_loading.nanomax_dataloader import (
    NanoMaxDataset,
    prepare_nanomax_data,
)
from ptycho_jax.transmission.background import (
    background_frame_counts,
    initialize_background,
)
from ptycho_jax.transmission.loss_and_reg import (
    gauss_loss,
    join_loss_and_forward_batched,
)
from ptycho_jax.transmission.optim_loop import run_optimization_loop
from ptycho_jax.transmission.optimizers import prepare_optimizer
from ptycho_jax.transmission.probe_initialization import (
    get_thresholded_intensity,
    init_probe_simple_fft,
)
from ptycho_jax.transmission.transmission_ptycho_models import (
    construct_point_model_ptycho_transmission,
)
from ptycho_jax.visualisation.probe_plotting import probe_display_geometry

DEFAULT_DATAPATH = Path(
    "/gpfs/exfel/data/scratch/kkonstantin/Projects/NanoMax/Test_reconstruction/"
    "download_data/09_kev/raw/0001_setup/000082.h5"
)


@dataclass(frozen=True)
class ReconstructionConfig:
    datapath: Path = DEFAULT_DATAPATH
    output_dir: Path = Path("results/nanomax")
    detector_distance_m: float = 7.180
    detector_pixel_size_m: float = 75e-6
    defocus_m: float = 1500e-6  # focus -> sample; positive downstream
    crop_shape: tuple[int, int] = (512, 512)
    crop_center: tuple[float, float] | None = None
    scan_coords_unit: float = 1e-6
    energy_unit: float = 1.0
    motor_x_sign: int = 1
    motor_y_sign: int = -1
    flip_detector_rows: bool = True
    scan_coord_limits: dict | None = None
    intensity_limits: tuple[float, float] | None = None
    max_positions: int | None = None
    n_steps: int = 400
    batch_size: int = 13
    shift_margin: int = 10
    max_correction_magnitude: float = 5.0
    object_padding: int = 32
    relative_probe_threshold: float = 5e-3
    sample_learning_rate: float = 0.03
    probe_learning_rate: float = 0.3
    weights_learning_rate: float = 0.003
    positions_learning_rate: float = 0.003
    seed: int = 0
    detector_name: str = "eiger4m"

    def __post_init__(self):
        NanoMaxDataset.detector_path_for_name(self.detector_name)
        if self.n_steps < 1 or self.batch_size < 1:
            raise ValueError("n_steps and batch_size must be positive.")
        if not 1 < self.max_correction_magnitude < self.shift_margin:
            raise ValueError("Require 1 < max_correction_magnitude < shift_margin.")
        if self.object_padding < 1:
            raise ValueError("object_padding must be positive.")
        if self.motor_x_sign not in (-1, 1) or self.motor_y_sign not in (-1, 1):
            raise ValueError("Motor signs must be -1 or +1.")
        if self.max_positions is not None and self.max_positions < 1:
            raise ValueError("max_positions must be positive when supplied.")


def load_reconstruction_data(config: ReconstructionConfig) -> dict:
    """Load once, apply this acquisition's orientation, and shift exactly once."""
    dataset = NanoMaxDataset(
        config.datapath,
        config.detector_distance_m,
        config.detector_pixel_size_m,
        config.defocus_m,
        detector_name=config.detector_name,
    )
    data = prepare_nanomax_data(
        dataset,
        crop_shape=config.crop_shape,
        crop_center=config.crop_center,
        scan_coords_unit=config.scan_coords_unit,
        energy_unit=config.energy_unit,
        scan_axis_order=(1, 0),
        fft_shift=False,
        scan_coord_limits=config.scan_coord_limits,
        intensity_limits=config.intensity_limits,
    )
    if config.max_positions is not None:
        for key in (
            "measured_intensities",
            "scan_coords",
            "total_ints",
            "position_id",
            "frame_indices",
        ):
            data[key] = data[key][: config.max_positions]
        _, first = np.unique(data["position_id"], return_index=True)
        data["mean_image"] = data["measured_intensities"][first].mean(axis=0)
        data["mean_probe"] = data["mean_image"].copy()
    if config.flip_detector_rows:
        for key in (
            "measured_intensities",
            "mean_image",
            "mean_probe",
            "mask",
            "precise_mask",
            "detector_mask",
        ):
            if data[key] is not None:
                data[key] = np.flip(data[key], axis=-2).copy()
    # The saved reference scan proves beam positions = (-motor_y, +motor_x)
    # up to offsets. Macro motor-flip flags also include a stage-to-beam sign.
    coords = data["scan_coords"][:, (1, 0)] * [config.motor_y_sign, config.motor_x_sign]
    pixels = coords * config.scan_coords_unit / data["reconstruction_pix_size"]
    pixels -= pixels.mean(axis=0)
    data["scan_coords_pix"] = pixels
    data["scan_axis_labels"] = (
        ("-" if config.motor_y_sign < 0 else "") + "sy (µm)",
        ("-" if config.motor_x_sign < 0 else "") + "sx (µm)",
    )
    data["int_parts"] = np.floor(pixels).astype(np.int64)
    data["float_parts"] = pixels - data["int_parts"]
    for key in (
        "measured_intensities",
        "mean_image",
        "mask",
        "precise_mask",
        "detector_mask",
    ):
        if data[key] is not None:
            data[key] = np.fft.ifftshift(data[key], axes=(-2, -1))
    if not np.all(data["total_ints"] > 0):
        raise ValueError(
            "Reconstruction requires positive total intensity in every retained frame."
        )
    return data


def initialize_reconstruction(
    data: dict, config: ReconstructionConfig, n_modes: int
) -> dict:
    """Construct parameters accepted by the current point model and optimizer."""
    if n_modes < 1:
        raise ValueError("n_modes must be positive.")
    geometry = dict(data["experimental_parameters"])
    # init_probe_simple_fft documents z_fs as focus -> sample, positive downstream.
    # Pass the physical NanoMAX defocus directly; the old MID sign reversal is absent.
    geometry["probe_initialization_z_fs_m"] = geometry["assumed_defocus_at_sample_m"]
    probe_intensity = get_thresholded_intensity(
        data["mean_probe"], config.relative_probe_threshold
    )
    probe, probe_pixel = init_probe_simple_fft(
        probe_intensity,
        geometry["detector_pixel_size_m"],
        geometry["sample_detector_distance_m"],
        geometry["probe_initialization_z_fs_m"],
        geometry["wavelength_m"],
    )
    if not np.isclose(
        probe_pixel, data["reconstruction_pix_size"][-1], rtol=1e-10, atol=0
    ):
        raise ValueError(
            "Probe initialization and detector crop have inconsistent sample pixels."
        )
    rng = np.random.default_rng(config.seed)
    modes = np.repeat(probe[None], n_modes, axis=0)
    if n_modes > 1:
        noise = rng.normal(size=modes.shape) + 1j * rng.normal(size=modes.shape)
        modes *= 1 + 0.01 * noise
    modes = modes.astype(np.complex64)
    mean_total = data["total_ints"].mean()
    far_field = np.fft.fft2(modes, norm="ortho", axes=(-2, -1))
    valid_energy = (np.abs(far_field) ** 2)[:, data["detector_mask"]].sum(dtype=float)
    if not np.isfinite(valid_energy) or valid_energy <= 0:
        raise ValueError(
            "Probe initialization has no finite energy on valid detector pixels."
        )
    modes *= np.sqrt(mean_total / valid_energy)
    # Weights multiply wave amplitudes, so their diagonal scales with sqrt(I/Imean).
    weights = (
        np.eye(n_modes, dtype=np.complex64)[None]
        * np.sqrt(
            data["total_ints"] / mean_total,
        )[:, None, None]
    )
    integer = data["int_parts"]
    probe_shape = np.asarray(data["mean_probe"].shape)
    positions = integer - integer.min(axis=0) + config.object_padding
    sample_shape = (
        np.ptp(integer, axis=0)
        + probe_shape
        + 2 * (config.shift_margin + config.object_padding)
    )
    # fourier_shift(patch, +shift) samples the object at r-shift. Thus -fraction
    # represents the requested floor(position)+fraction sampling coordinate.
    raw_shifts = np.arctanh(-data["float_parts"] / config.max_correction_magnitude)
    params = {
        "sample": jnp.ones(tuple(sample_shape), dtype=jnp.complex64),
        "probe_modes": jnp.asarray(modes, dtype=jnp.complex64),
        "modal_weights": jnp.asarray(weights, dtype=jnp.complex64),
        "scan_mistakes": jnp.asarray(raw_shifts, dtype=jnp.float32),
    }
    fixed = {"sample_positions": jnp.asarray(positions, dtype=jnp.int32)}
    background_enabled = getattr(config, "background", False)
    if background_enabled:
        background_params, background_fixed = initialize_background(
            data["detector_mask"],
            data["total_ints"],
            config.background_grid_shape,
            config.background_initial_fraction,
            config.background_max_fraction,
        )
        params.update(background_params)
        fixed.update(background_fixed)
    model_parameters = {
        "probe_type": "fluctuating",
        "sample_type": "complex",
        "binning_factor": (1, 1),
        "sample_selector_type": "correcting",
        "propagator_type": "Fourrier",
        "probe_shape": tuple(probe_shape),
        "max_correction_magnitude": config.max_correction_magnitude,
        "shift_margin": config.shift_margin,
        "background_type": "shared_scaled" if background_enabled else "none",
    }
    models = construct_point_model_ptycho_transmission(model_parameters)
    predict_batch = jax.jit(jax.vmap(models["point_model"], in_axes=(None, None, 0)))
    # The shared selector tapers an enlarged object patch before shifting it.
    # Fractional interpolation changes blank-sample energy slightly. Calibrate
    # amplitudes against that exact model, including its mask and selector.
    predicted_totals = []
    for start in range(0, len(weights), config.batch_size):
        stop = min(start + config.batch_size, len(weights))
        predicted = np.asarray(
            predict_batch(params, fixed, jnp.arange(start, stop, dtype=jnp.int32))
        )
        predicted_totals.extend(
            predicted[:, data["detector_mask"]].sum(axis=1, dtype=float)
        )
    predicted_totals = np.asarray(predicted_totals)
    if not np.isfinite(predicted_totals).all() or np.any(predicted_totals <= 0):
        raise ValueError("Initial forward model has invalid frame energies.")
    background_counts = (
        np.asarray(background_frame_counts(params, fixed), dtype=float)
        if background_enabled
        else 0.0
    )
    signal_totals = predicted_totals - background_counts
    if np.any(signal_totals <= 0):
        raise ValueError(
            "Initial diffraction signal must have positive frame energies."
        )
    params["modal_weights"] *= jnp.asarray(
        np.sqrt((data["total_ints"] - background_counts) / signal_totals)[
            :, None, None
        ],
        dtype=jnp.float32,
    )
    return {
        "params": params,
        "fixed": fixed,
        "models": models,
        "predict_batch": predict_batch,
        "mask": jnp.asarray(data["detector_mask"], dtype=bool),
        "root_measured": np.sqrt(data["measured_intensities"]),
        "probe_intensity": probe_intensity,
        "probe_pixel_size_m": probe_pixel,
        "geometry": geometry,
        "object_pixel_origin": config.object_padding
        + config.shift_margin
        - integer.min(axis=0)
        + probe_shape / 2,
    }


def evaluate_reconstruction(params, fixed, state: dict, batch_size: int) -> dict:
    """Evaluate every frame, including a partial final batch, on valid pixels."""
    predict = state["predict_batch"]
    root = state["root_measured"]
    mask = np.asarray(state["mask"])
    frame_losses, predicted_totals = [], []
    mean_prediction = np.zeros(mask.shape, dtype=np.float64)
    first_prediction = None
    for start in range(0, len(root), batch_size):
        stop = min(start + batch_size, len(root))
        prediction = np.asarray(
            predict(params, fixed, jnp.arange(start, stop, dtype=jnp.int32))
        )
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite reconstructed intensity.")
        residual = np.sqrt(np.maximum(prediction, 0)) - root[start:stop]
        frame_losses.extend(np.mean(residual[:, mask] ** 2, axis=1, dtype=np.float64))
        predicted_totals.extend(prediction[:, mask].sum(axis=1, dtype=np.float64))
        mean_prediction += prediction.sum(axis=0, dtype=np.float64)
        if first_prediction is None:
            first_prediction = prediction[0]
    return {
        "loss": float(np.mean(frame_losses)),
        "frame_losses": np.asarray(frame_losses),
        "predicted_totals": np.asarray(predicted_totals),
        "mean_prediction": mean_prediction / len(root),
        "first_prediction": first_prediction,
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _save_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n"
    )


def _save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_preparation_figure(data: dict, output: Path, filename_prefix: str = ""):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    xy = data["scan_coords_pix"] * data["reconstruction_pix_size"] / 1e-6
    scatter = axes[0, 0].scatter(xy[:, 1], xy[:, 0], c=data["total_ints"], s=20)
    axes[0, 0].set(
        xlabel=data["scan_axis_labels"][1],
        ylabel=data["scan_axis_labels"][0],
        title="Scan positions / total counts",
    )
    axes[0, 0].set_aspect("equal")
    fig.colorbar(scatter, ax=axes[0, 0])
    for ax, key, title in (
        (axes[0, 1], "mean_image", "Brightest-per-position mean: log10(1+I)"),
        (axes[1, 0], "mean_probe", "Probe initialization image: log10(1+I)"),
    ):
        value = data[key] if key == "mean_probe" else np.fft.fftshift(data[key])
        fig.colorbar(ax.imshow(np.log10(1 + value), cmap="magma"), ax=ax)
        ax.set_title(title)
    axes[0, 2].imshow(np.fft.fftshift(data["detector_mask"]), cmap="gray")
    axes[0, 2].set_title("Valid detector mask after orientation")
    axes[1, 1].imshow(
        np.log10(1 + np.fft.fftshift(data["measured_intensities"][0])), cmap="magma"
    )
    axes[1, 1].set_title("First measured frame: log10(1+I)")
    axes[1, 2].plot(data["total_ints"])
    axes[1, 2].set(
        xlabel="Retained frame",
        ylabel="Counts",
        title="Masked cropped intensity totals",
    )
    _save_figure(fig, output / f"{filename_prefix}preparation.png")


def _save_probe(modes, path: Path, title: str, pixel_size_m=None):
    modes = np.asarray(modes)
    extent, xlabel, ylabel, size_label = probe_display_geometry(
        modes.shape[-2:], pixel_size_m
    )
    fig, axes = plt.subplots(
        2, len(modes), figsize=(max(5, 3 * len(modes)), 6), squeeze=False
    )
    for i, mode in enumerate(modes):
        image = axes[0, i].imshow(
            np.abs(mode), cmap="viridis", extent=extent, origin="lower"
        )
        fig.colorbar(image, ax=axes[0, i], shrink=0.7)
        axes[0, i].set_title(f"Mode {i + 1} amplitude")
        axes[1, i].imshow(
            np.angle(mode),
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            extent=extent,
            origin="lower",
        )
        axes[1, i].set_title("Phase (rad)")
        for row in range(2):
            axes[row, i].set(xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(title + "\n" + size_label)
    _save_figure(fig, path)


def _save_reconstruction_figures(
    data,
    state,
    params,
    initial,
    final,
    history,
    folder,
    config,
    filename_prefix="",
    objective_label="Masked amplitude MSE",
):
    sample = np.asarray(params["sample"])
    coverage = np.zeros(sample.shape)
    power = np.sum(np.abs(np.asarray(params["probe_modes"])) ** 2, axis=0)
    for corner in np.asarray(state["fixed"]["sample_positions"]):
        y, x = corner + config.shift_margin
        coverage[y : y + power.shape[0], x : x + power.shape[1]] += power
    supported = coverage > coverage.max() * 0.01
    ypix, xpix = np.nonzero(supported)
    roi = (slice(ypix.min(), ypix.max() + 1), slice(xpix.min(), xpix.max() + 1))
    dy, dx = data["reconstruction_pix_size"] / 1e-6
    oy, ox = state["object_pixel_origin"]
    extent = (
        (roi[1].start - ox) * dx,
        (roi[1].stop - ox) * dx,
        (roi[0].start - oy) * dy,
        (roi[0].stop - oy) * dy,
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, image, title, cmap in (
        (axes[0], np.abs(sample), "Object amplitude", "gray"),
        (
            axes[1],
            np.angle(sample),
            "Object phase (rad; arbitrary global phase)",
            "twilight",
        ),
        (
            axes[2],
            coverage / coverage.max(),
            "Approximate illumination coverage",
            "magma",
        ),
    ):
        image = np.where(supported, image, np.nan)
        if cmap == "twilight":
            phase_low, phase_high = np.percentile(image[supported], (1, 99))
            limits = (
                dict(vmin=phase_low, vmax=phase_high) if phase_high > phase_low else {}
            )
        else:
            limits = {}
        artist = ax.imshow(
            image[roi], cmap=cmap, extent=extent, origin="lower", **limits
        )
        ax.set(title=title, xlabel="x (µm)", ylabel="y (µm)")
        fig.colorbar(artist, ax=ax, shrink=0.8)
    _save_figure(fig, folder / f"{filename_prefix}sample.png")
    _save_probe(
        state["params"]["probe_modes"],
        folder / f"{filename_prefix}probe_initial.png",
        "Initial probe modes",
        pixel_size_m=data["reconstruction_pix_size"],
    )
    _save_probe(
        params["probe_modes"],
        folder / f"{filename_prefix}probe_final.png",
        "Reconstructed probe modes",
        pixel_size_m=data["reconstruction_pix_size"],
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(len(history)), history, label="Before each epoch update")
    ax.scatter(
        [len(history)],
        [final["loss"]],
        marker="x",
        color="red",
        label="Final evaluation",
    )
    ax.set(xlabel="Epoch", ylabel=objective_label, title="Reconstruction objective")
    ax.legend()
    _save_figure(fig, folder / f"{filename_prefix}loss.png")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    images = [
        data["measured_intensities"][0],
        initial["first_prediction"],
        final["first_prediction"],
        data["measured_intensities"].mean(axis=0),
        final["mean_prediction"],
        np.abs(np.sqrt(final["first_prediction"]) - state["root_measured"][0]),
    ]
    titles = [
        "First measured",
        "First predicted: initial",
        "First predicted: final",
        "Mean measured",
        "Mean predicted: final",
        "First-frame amplitude residual",
    ]
    for i, (ax, image, title) in enumerate(zip(axes.flat, images, titles)):
        value = image if i == 5 else np.log10(1 + image)
        value = np.where(data["detector_mask"], value, np.nan)
        artist = ax.imshow(np.fft.fftshift(value), cmap="magma")
        ax.set_title(title + ("" if i == 5 else " [log10(1+I)]"))
        fig.colorbar(artist, ax=ax, shrink=0.8)
    _save_figure(fig, folder / f"{filename_prefix}diffraction.png")
    return coverage


def run_reconstruction(data: dict, config: ReconstructionConfig, n_modes: int) -> dict:
    folder = Path(config.output_dir) / f"modes_{n_modes}"
    folder.mkdir(parents=True, exist_ok=False)
    state = initialize_reconstruction(data, config, n_modes)
    params, fixed = state["params"], state["fixed"]
    initial = evaluate_reconstruction(params, fixed, state, config.batch_size)
    print(
        f"Modes={n_modes}: {len(data['total_ints'])} frames; initial loss={initial['loss']:.6g}",
        flush=True,
    )
    loss = jax.jit(
        join_loss_and_forward_batched(gauss_loss, state["models"]["point_model"])
    )
    loss_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0))
    rates = {
        "sample": config.sample_learning_rate,
        "probe_modes": config.probe_learning_rate,
        "modal_weights": config.weights_learning_rate,
        "scan_mistakes": config.positions_learning_rate,
    }
    optimizer = prepare_optimizer(
        {key: {"type": "adam", "learning_rate": lr} for key, lr in rates.items()}
    )
    start = time.perf_counter()
    params, fixed, _, history = run_optimization_loop(
        params=params,
        non_diff_params=fixed,
        opt_state=optimizer.init(params),
        optimizer=optimizer,
        loss_and_grad_fn=loss_and_grad,
        measured_batch_pool=state["root_measured"],
        mask=state["mask"],
        mode="accumulate_full_pass",
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle_each_epoch=True,
        use_multigpu=False,
    )
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - start
    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(params)):
        raise FloatingPointError("Nonfinite reconstruction parameters.")
    final = evaluate_reconstruction(params, fixed, state, config.batch_size)
    history = np.asarray(history)
    if not np.isfinite(history).all():
        raise FloatingPointError("Nonfinite reconstruction loss history.")
    corrections = (
        -(
            np.tanh(np.asarray(params["scan_mistakes"]))
            - np.tanh(np.asarray(state["params"]["scan_mistakes"]))
        )
        * config.max_correction_magnitude
    )
    mean_measured = data["total_ints"].mean() / data["detector_mask"].sum()
    metrics = {
        "modes": n_modes,
        "positions": len(data["total_ints"]),
        "epochs": config.n_steps,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "initial_loss": initial["loss"],
        "final_loss": final["loss"],
        "loss_reduction_percent": 100 * (1 - final["loss"] / initial["loss"]),
        "initial_relative_amplitude_rmse": float(
            np.sqrt(initial["loss"] / mean_measured)
        ),
        "final_relative_amplitude_rmse": float(np.sqrt(final["loss"] / mean_measured)),
        "optimization_seconds_including_compile": elapsed,
        "position_correction_rms_pixels": float(np.sqrt(np.mean(corrections**2))),
        "position_correction_max_pixels": float(np.abs(corrections).max()),
        "geometry": state["geometry"],
        "sample_pixel_size_m": data["reconstruction_pix_size"],
        "object_shape": np.asarray(params["sample"].shape),
        "detector_center_raw": data["detector_center"],
        "detector_crop_raw": [[s.start, s.stop] for s in data["crop_slices"]],
        "valid_detector_fraction": float(data["detector_mask"].mean()),
        "device": str(params["sample"].device),
        "device_kind": params["sample"].device.device_kind,
        "jax_version": jax.__version__,
    }
    coverage = _save_reconstruction_figures(
        data, state, params, initial, final, history, folder, config
    )
    np.savez_compressed(
        folder / "reconstruction.npz",
        **jax.device_get(params),
        sample_positions=np.asarray(fixed["sample_positions"]),
        scan_coords=data["scan_coords"],
        scan_coords_pix=data["scan_coords_pix"],
        frame_indices=data["frame_indices"],
        position_id=data["position_id"],
        int_parts=data["int_parts"],
        float_parts=data["float_parts"],
        position_corrections_pix=corrections,
        object_pixel_origin=state["object_pixel_origin"],
        loss_history=history,
        detector_mask=data["detector_mask"],
        total_ints=data["total_ints"],
        illumination_coverage=coverage,
        sample_pixel_size_m=data["reconstruction_pix_size"],
        initial_frame_losses=initial["frame_losses"],
        final_frame_losses=final["frame_losses"],
        predicted_totals=final["predicted_totals"],
        measured_first=data["measured_intensities"][0],
        predicted_first=final["first_prediction"],
        predicted_mean=final["mean_prediction"],
        initial_probe=np.asarray(state["params"]["probe_modes"]),
    )
    _save_json(folder / "metrics.json", metrics)
    print(
        f"Modes={n_modes}: final loss={final['loss']:.6g}; reduction={metrics['loss_reduction_percent']:.2f}%; "
        f"optimization={elapsed:.1f}s; saved {folder}",
        flush=True,
    )
    return metrics


def save_run_report(data, config, metrics):
    output = Path(config.output_dir)
    config_data = asdict(config)
    config_data["created_utc"] = datetime.now(timezone.utc).isoformat()
    config_data["source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("pipeline.py", "pipeline_nanomax.py", "pipeline_nanomax_upd.py")
    }
    _save_json(output / "configuration.json", config_data)
    _save_json(output / "summary.json", metrics)
    fig, ax = plt.subplots(figsize=(7, 4))
    for result in metrics:
        n = result["modes"]
        with np.load(output / f"modes_{n}" / "reconstruction.npz") as archive:
            history = archive["loss_history"]
        ax.semilogy(
            np.arange(len(history) + 1),
            np.r_[history, result["final_loss"]],
            label=f"{n} mode(s)",
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Masked amplitude MSE",
        title="NanoMAX reconstruction comparison",
    )
    ax.legend()
    _save_figure(fig, output / "comparison.png")
    geometry = metrics[0]["geometry"]
    report = [
        "# NanoMAX reconstruction results",
        "",
        f"Input: `{config.datapath}`.",
        "",
        f"Energy: {geometry['energy_eV']:.9f} eV; wavelength: {geometry['wavelength_m']:.12g} m.",
        f"Detector distance: {geometry['sample_detector_distance_m']:.6f} m; "
        f"detector pixels: {geometry['detector_pixel_size_m']:.9g} m; "
        f"focus-to-sample defocus passed to initializer: {geometry['probe_initialization_z_fs_m']:+.6f} m.",
        f"All {len(data['total_ints'])} retained frames use crop {config.crop_shape}; "
        f"sample pixels are {data['reconstruction_pix_size'] * 1e9} nm.",
        "",
        "| Modes | Epochs | Initial MSE | Final MSE | Reduction | Final relative amplitude RMSE | Optimization time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in metrics:
        report.append(
            f"| {item['modes']} | {item['epochs']} | {item['initial_loss']:.6g} | "
            f"{item['final_loss']:.6g} | {item['loss_reduction_percent']:.2f}% | "
            f"{item['final_relative_amplitude_rmse']:.4%} | "
            f"{item['optimization_seconds_including_compile']:.1f} |"
        )
    report += [
        "",
        "![Preparation](preparation.png)",
        "",
        "![Loss comparison](comparison.png)",
        "",
    ]
    for item in metrics:
        n = item["modes"]
        report += [
            f"## {n} probe mode(s)",
            "",
            f"![Object](modes_{n}/sample.png)",
            "",
            f"![Initial probe](modes_{n}/probe_initial.png)",
            "",
            f"![Final probe](modes_{n}/probe_final.png)",
            "",
            f"![Diffraction fit](modes_{n}/diffraction.png)",
            "",
            f"Arrays: [reconstruction.npz](modes_{n}/reconstruction.npz); "
            f"metrics: [metrics.json](modes_{n}/metrics.json).",
            "",
        ]
    report += [
        "## Interpretation and limits",
        "",
        "These are training-data fits, without a held-out scan or ground truth. "
        "A lower loss alone does not establish spatial resolution or uniqueness. "
        "Object/probe scale, phase and phase-ramp ambiguities remain. "
        "Object phase colors span the 1st to 99th percentiles inside the displayed coverage. "
        "The probe is initialized from the mean sample diffraction, since no empty-beam image was supplied. "
        f"Motor scale: {config.scan_coords_unit:g} m per unit; array-axis beam positions use "
        f"{data['scan_axis_labels']}. Detector row flip: {config.flip_detector_rows}; "
        "the reference macro uses slightly different pseudo/x and pseudo/y values. "
        "The displayed object is masked below 1% of approximate illumination coverage; "
        "this coverage ignores fractional shifts and position-dependent modal weights. "
        "Defocus and distance are supplied acquisition settings, not fitted calibrations. "
        "The detector crop limits the acquired spatial frequencies; no full-detector resolution claim is made.",
        "",
    ]
    (output / "results.md").write_text("\n".join(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", type=Path, default=DEFAULT_DATAPATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=13)
    parser.add_argument("--crop-shape", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--crop-center", type=float, nargs=2)
    parser.add_argument("--detector-distance-m", type=float, default=7.180)
    parser.add_argument("--detector-pixel-size-m", type=float, default=75e-6)
    parser.add_argument("--defocus-m", type=float, default=1500e-6)
    parser.add_argument("--scan-coords-unit", type=float, default=1e-6)
    parser.add_argument("--energy-unit", type=float, default=1.0)
    parser.add_argument("--motor-x-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--motor-y-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument(
        "--flip-detector-rows", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    args = parser.parse_args(argv)
    if len(set(args.modes)) != len(args.modes) or min(args.modes) < 1:
        parser.error("--modes must contain distinct positive mode counts.")
    output = args.output_dir or Path("results/nanomax") / (
        f"scan_{args.datapath.stem}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if output.exists():
        parser.error(
            f"Output directory already exists: {output}; choose a new directory."
        )
    config = ReconstructionConfig(
        datapath=args.datapath,
        output_dir=output,
        detector_distance_m=args.detector_distance_m,
        detector_pixel_size_m=args.detector_pixel_size_m,
        defocus_m=args.defocus_m,
        crop_shape=tuple(args.crop_shape),
        crop_center=None if args.crop_center is None else tuple(args.crop_center),
        scan_coords_unit=args.scan_coords_unit,
        energy_unit=args.energy_unit,
        motor_x_sign=args.motor_x_sign,
        motor_y_sign=args.motor_y_sign,
        flip_detector_rows=args.flip_detector_rows,
        max_positions=args.max_positions,
        n_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    device = jax.devices(args.device)[0] if args.device else jax.devices()[0]
    with jax.default_device(device):
        print(f"Loading {config.datapath} on {device}", flush=True)
        data = load_reconstruction_data(config)
        print(f"Geometry: {data['experimental_parameters']}", flush=True)
        output.mkdir(parents=True)
        save_preparation_figure(data, output)
        _save_json(output / "configuration.json", asdict(config))
        results = [run_reconstruction(data, config, n_modes) for n_modes in args.modes]
        save_run_report(data, config, results)
        print(f"Report: {output / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
