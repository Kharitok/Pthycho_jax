r"""NanoMAX AD with scheduled TV/projections and full beam diagnostics.

Examples (output-dir is a parent directory; every run creates a fresh child)::

    PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
        --config configs/nanomax_scheduled.json
    PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
        --modes 5 --steps 400 --sample-tv-weight 1 --tag reg_test --device gpu
    PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
        --modes 5 --steps 400 --probe-support --tag 'projection test' --device gpu

The historical entry point retains its defaults. Loading, geometry, initialization
and figures are shared with it. Both constraints can be enabled together and
scheduled independently using INDEX or START:STOP[:EVERY] (zero-based, STOP
exclusive). Diagnostics run at the end unless --no-diagnostics is passed. Python
callers can additionally supply regularizer_fn(params, fixed) -> scalar and
projection_fn(params, fixed) -> (params, fixed) to run_pipeline/run_reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import secrets
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Import first: the original pipeline configures GPU allocation and plotting.
import pipeline_nanomax_upd as base  # isort: skip
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ptycho_jax.transmission.background import audit_background, background_components
from ptycho_jax.transmission.loss_and_reg import (
    combine_regularizers,
    create_background_regularizer,
    create_parameter_tv_regularizer,
    gauss_loss,
    join_loss_and_forward_batched,
    join_loss_and_reg,
)
from ptycho_jax.transmission.optim_loop import run_optimization_loop
from ptycho_jax.transmission.optimizers import prepare_optimizer
from ptycho_jax.transmission.projectors import (
    create_smooth_spatial_support,
    create_spatial_support_projection,
)
from ptycho_jax.transmission.schedules import (
    STEP_KEY,
    schedule_constraints,
    schedule_mask,
)
from ptycho_jax.transmission.settings import load_json_settings, validate_setting_values
from ptycho_jax.visualisation import nanomax_diagnostics
from ptycho_jax.visualisation.background_diagnostics import save_background_figures


@dataclass(frozen=True)
class ReconstructionConfig(base.ReconstructionConfig):
    """One run's settings and identity, captured before loading/initializing.

    started_utc is an aware ISO-8601 time, saved at full precision; names use UTC
    date and time to seconds. Spaces in tags become underscores in filenames;
    the original tag is retained in configuration/metrics. Default tags contain
    five random ASCII letters, independent of the reconstruction seed.
    """

    tag: str = field(
        default_factory=lambda: "".join(
            secrets.choice(string.ascii_letters) for _ in range(5)
        )
    )
    started_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    sample_tv_weight: float = 0.0
    probe_tv_weight: float = 0.0
    tv_p: float = 2.0
    tv_q: float = 1.0
    tv_epsilon: float = 1e-6
    tv_normalize: bool = True
    probe_support: bool = False
    probe_support_radius_px: float | None = None
    probe_support_decay_px: float | None = None
    probe_support_center_px: tuple[float, float] | None = None
    regularization_schedule: tuple[str, ...] | None = None
    projection_schedule: tuple[str, ...] | None = None
    diagnostics: bool = True
    half_range_m: float | None = None
    z_samples: int = 201
    padding_factor: int = 3
    background: bool = False
    background_grid_shape: tuple[int, int] = (16, 16)
    background_initial_fraction: float = 1e-4
    background_max_fraction: float = 0.05
    background_learning_rate: float = 0.03
    background_scale_learning_rate: float = 0.03
    background_start_step: int = 100
    background_l1_weight: float = 0.01
    background_smoothness_weight: float = 1.0
    background_scale_weight: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if not (
            math.isfinite(self.background_initial_fraction)
            and math.isfinite(self.background_max_fraction)
            and 0 < self.background_initial_fraction < self.background_max_fraction < 1
        ):
            raise ValueError(
                "Require 0 < background_initial_fraction < background_max_fraction < 1."
            )
        if len(self.background_grid_shape) != 2 or any(
            type(n) is not int or n < 2 for n in self.background_grid_shape
        ):
            raise ValueError("background_grid_shape must contain two integers >=2.")
        if self.background and any(
            a > b
            for a, b in zip(self.background_grid_shape, self.crop_shape, strict=True)
        ):
            raise ValueError("Background grid must not exceed detector crop shape.")
        if (
            type(self.background_start_step) is not int
            or self.background_start_step < 0
        ):
            raise ValueError("background_start_step must be a nonnegative integer.")
        for name in (
            "background_learning_rate",
            "background_scale_learning_rate",
            "background_l1_weight",
            "background_smoothness_weight",
            "background_scale_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        schedule_mask(self.regularization_schedule, self.n_steps)
        schedule_mask(self.projection_schedule, self.n_steps)
        if (
            not isinstance(self.z_samples, int)
            or self.z_samples < 5
            or self.z_samples % 2 != 1
        ):
            raise ValueError("z_samples must be an odd integer >= 5.")
        if not isinstance(self.padding_factor, int) or self.padding_factor < 1:
            raise ValueError("padding_factor must be a positive integer.")
        half_range = (
            2 * abs(self.defocus_m) if self.half_range_m is None else self.half_range_m
        )
        if (self.diagnostics or self.half_range_m is not None) and (
            not math.isfinite(half_range) or half_range <= 0
        ):
            raise ValueError(
                "Propagation half_range_m must be finite and positive; supply it explicitly for zero defocus."
            )
        if (
            not isinstance(self.tag, str)
            or not self.tag.strip()
            or any(c in self.tag for c in "/\\")
        ):
            raise ValueError("tag must be nonempty and contain no path separators.")
        if not self.tag_slug:
            raise ValueError(
                "tag must contain ASCII letters, digits, underscores or hyphens."
            )
        start = datetime.fromisoformat(self.started_utc)
        if start.utcoffset() is None:
            raise ValueError("started_utc must include a timezone.")
        for name in ("sample_tv_weight", "probe_tv_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in (
            "tv_p",
            "tv_q",
            "tv_epsilon",
            "probe_support_radius_px",
            "probe_support_decay_px",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        if self.probe_support_center_px is not None and (
            len(self.probe_support_center_px) != 2
            or not all(math.isfinite(v) for v in self.probe_support_center_px)
        ):
            raise ValueError(
                "probe_support_center_px must contain two finite coordinates."
            )

    @property
    def tag_slug(self):
        return re.sub(r"[^A-Za-z0-9_-]+", "_", self.tag.strip()).strip("_")

    @property
    def run_id(self):
        stamp = (
            datetime.fromisoformat(self.started_utc)
            .astimezone(UTC)
            .strftime("%Y%m%dT%H%M%SZ")
        )
        return f"{stamp}_{self.tag_slug}"

    @property
    def run_dir(self):
        return Path(self.output_dir) / f"scan_{Path(self.datapath).stem}_{self.run_id}"

    def artifact(self, name: str, folder: Path | None = None):
        return (
            self.run_dir if folder is None else Path(folder)
        ) / f"{self.run_id}_{name}"


def _callback_name(callback):
    return (
        None
        if callback is None
        else f"{callback.__module__}.{getattr(callback, '__qualname__', type(callback).__qualname__)}"
    )


def run_reconstruction(
    data, config, n_modes, *, regularizer_fn=None, projection_fn=None
):
    """Run full-pass AD; apply constraints on independently scheduled updates.

    Regularization is attached to the mean batch data loss before gradient
    accumulation. Batch-size weighting therefore contributes one regularizer
    per full pass, rather than scaling it with the number of batches.
    A custom projection runs before the configured probe support.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be positive.")
    folder = config.run_dir / f"modes_{n_modes}"
    folder.mkdir(parents=True, exist_ok=False)
    state = base.initialize_reconstruction(data, config, n_modes)
    params, fixed = state["params"], state["fixed"]
    fixed = {**fixed, STEP_KEY: jnp.array(0, dtype=jnp.int32)}
    initial = base.evaluate_reconstruction(params, fixed, state, config.batch_size)
    regularizers = [
        create_parameter_tv_regularizer(
            {"sample": config.sample_tv_weight, "probe_modes": config.probe_tv_weight},
            p=config.tv_p,
            q=config.tv_q,
            epsilon=config.tv_epsilon,
            normalize=config.tv_normalize,
        )
    ]
    if config.background:
        regularizers.append(
            create_background_regularizer(
                config.background_l1_weight,
                config.background_smoothness_weight,
                config.background_scale_weight,
            )
        )
    if regularizer_fn is not None:
        regularizers.append(regularizer_fn)
    regularizer = jax.jit(combine_regularizers(regularizers))
    initial_configured_reg = float(regularizer(params, fixed))

    support = None
    support_info = None
    projections = [] if projection_fn is None else [projection_fn]
    if config.probe_support:
        shape = params["probe_modes"].shape[-2:]
        radius = config.probe_support_radius_px or 0.4 * min(shape)
        decay = config.probe_support_decay_px or 0.1 * min(shape)
        center = config.probe_support_center_px or tuple((n - 1) / 2 for n in shape)
        support = create_smooth_spatial_support(shape, radius, decay, center)
        if not np.any(np.asarray(support) > 0):
            raise ValueError("Probe support excludes every pixel.")
        projections.append(create_spatial_support_projection(support))
        support_info = {
            "type": "multiplicative_cosine_taper",
            "radius_px": radius,
            "decay_px": decay,
            "zero_from_radius_px": radius + decay,
            "center_yx_px": center,
            "radius_yx_m": radius * data["reconstruction_pix_size"],
            "decay_yx_m": decay * data["reconstruction_pix_size"],
            "initial_probe_power_retained_after_one_application": float(
                np.sum(
                    np.abs(np.asarray(params["probe_modes"]) * np.asarray(support)) ** 2
                )
                / np.sum(np.abs(np.asarray(params["probe_modes"])) ** 2)
            ),
        }

    @jax.jit
    def project(parameters, non_diff):
        for projection in projections:
            parameters, non_diff = projection(parameters, non_diff)
        return parameters, non_diff

    reg_active = schedule_mask(config.regularization_schedule, config.n_steps) & bool(
        config.sample_tv_weight > 0
        or config.probe_tv_weight > 0
        or regularizer_fn is not None
        or (
            config.background
            and any(
                w > 0
                for w in (
                    config.background_l1_weight,
                    config.background_smoothness_weight,
                    config.background_scale_weight,
                )
            )
        )
    )
    projection_active = schedule_mask(
        config.projection_schedule, config.n_steps
    ) & bool(projections)
    scheduled_reg, scheduled_project = schedule_constraints(
        regularizer,
        project,
        reg_active,
        projection_active,
    )
    initial_reg = initial_configured_reg if reg_active[0] else 0.0
    data_loss = join_loss_and_forward_batched(
        gauss_loss, state["models"]["point_model"]
    )
    raw_loss_and_grad = jax.value_and_grad(join_loss_and_reg(data_loss, scheduled_reg))

    @jax.jit
    def loss_and_grad(parameters, non_diff, indices, measured, mask):
        value, gradients = raw_loss_and_grad(
            parameters, non_diff, indices, measured, mask
        )
        if config.background:
            active = non_diff[STEP_KEY] >= config.background_start_step
            for name in ("background_logits", "background_scale_logits"):
                gradients[name] = jnp.where(
                    active, gradients[name], jnp.zeros_like(gradients[name])
                )
        return value, gradients

    rates = {
        "sample": config.sample_learning_rate,
        "probe_modes": config.probe_learning_rate,
        "modal_weights": config.weights_learning_rate,
        "scan_mistakes": config.positions_learning_rate,
    }
    if config.background:
        rates.update(
            background_logits=config.background_learning_rate,
            background_scale_logits=config.background_scale_learning_rate,
        )
    optimizer = prepare_optimizer(
        {key: {"type": "adam", "learning_rate": lr} for key, lr in rates.items()}
    )
    print(
        f"Modes={n_modes}: {len(data['total_ints'])} frames; initial data MSE={initial['loss']:.6g}; regularization={initial_reg:.6g}",
        flush=True,
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
        projection_fn=scheduled_project,
    )
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - start
    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(params)):
        raise FloatingPointError("Nonfinite reconstruction parameters.")
    final = base.evaluate_reconstruction(params, fixed, state, config.batch_size)
    final_configured_reg = float(regularizer(params, fixed))
    final_reg = final_configured_reg if reg_active[-1] else 0.0
    history = np.asarray(history)
    if not np.isfinite(history).all() or not np.isfinite(final_configured_reg):
        raise FloatingPointError("Nonfinite reconstruction objective.")
    corrections = (
        -(
            np.tanh(np.asarray(params["scan_mistakes"]))
            - np.tanh(np.asarray(state["params"]["scan_mistakes"]))
        )
        * config.max_correction_magnitude
    )
    mean_measured = data["total_ints"].mean() / data["detector_mask"].sum()
    metrics = {
        "run_id": config.run_id,
        "tag": config.tag,
        "started_utc": config.started_utc,
        "modes": n_modes,
        "positions": len(data["total_ints"]),
        "epochs": config.n_steps,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "initial_loss": initial["loss"],
        "final_loss": final["loss"],
        "initial_regularization": initial_reg,
        "final_regularization": final_reg,
        "initial_configured_regularization": initial_configured_reg,
        "final_configured_regularization": final_configured_reg,
        "final_objective_convention": "final parameters with last update's regularization setting",
        "regularization_schedule": config.regularization_schedule,
        "projection_schedule": config.projection_schedule,
        "regularization_active_steps": int(reg_active.sum()),
        "projection_active_steps": int(projection_active.sum()),
        "final_update_regularization_active": bool(reg_active[-1]),
        "final_update_projection_active": bool(projection_active[-1]),
        "optimization_steps_completed": int(fixed[STEP_KEY]),
        "initial_objective": initial["loss"] + initial_reg,
        "final_objective": final["loss"] + final_reg,
        "loss_history_kind": "data_mse_plus_regularization_before_each_update",
        "loss_reduction_percent": 100 * (1 - final["loss"] / initial["loss"]),
        "initial_relative_amplitude_rmse": float(
            np.sqrt(initial["loss"] / mean_measured)
        ),
        "final_relative_amplitude_rmse": float(np.sqrt(final["loss"] / mean_measured)),
        "optimization_seconds_including_compile": elapsed,
        "position_correction_rms_pixels": float(np.sqrt(np.mean(corrections**2))),
        "position_correction_max_pixels": float(np.abs(corrections).max()),
        "sample_tv_weight": config.sample_tv_weight,
        "probe_tv_weight": config.probe_tv_weight,
        "tv_p": config.tv_p,
        "tv_q": config.tv_q,
        "tv_epsilon": config.tv_epsilon,
        "tv_normalize": config.tv_normalize,
        "custom_regularizer": _callback_name(regularizer_fn),
        "custom_projection": _callback_name(projection_fn),
        "probe_support": support_info,
        "geometry": state["geometry"],
        "detector_source": data.get("detector_source"),
        "sample_pixel_size_m": data["reconstruction_pix_size"],
        "object_shape": np.asarray(params["sample"].shape),
        "detector_center_raw": data["detector_center"],
        "detector_crop_raw": [[s.start, s.stop] for s in data["crop_slices"]],
        "valid_detector_fraction": float(data["detector_mask"].mean()),
        "device": str(params["sample"].device),
        "device_kind": params["sample"].device.device_kind,
        "jax_version": jax.__version__,
    }
    background_arrays = {}
    if config.background:
        components, metrics["background"] = audit_background(
            params,
            fixed,
            data["measured_intensities"].mean(axis=0),
            final["mean_prediction"],
            final["predicted_totals"],
        )
        metrics["background"].update(
            grid_shape=config.background_grid_shape,
            learning_start_step=config.background_start_step,
        )
        initial_components = background_components(state["params"], state["fixed"])
        background_arrays = {
            "background_map": components["map"],
            "background_scales": components["scales"],
            "background_frame_counts": components["frame_counts"],
            "background_fractions": components["fractions"],
            "initial_background_map": initial_components["map"],
            "initial_background_scale_logits": np.asarray(
                state["params"]["background_scale_logits"]
            ),
        }
        save_background_figures(
            components,
            data,
            final,
            folder,
            config.run_id + "_",
            config.background_max_fraction,
        )
    else:
        metrics["background"] = {"enabled": False}
    # Only the loss marker consumes this replacement; diffraction arrays stay intact.
    coverage = base._save_reconstruction_figures(
        data,
        state,
        params,
        initial,
        {**final, "loss": metrics["final_objective"]},
        history,
        folder,
        config,
        filename_prefix=config.run_id + "_",
        objective_label="Amplitude MSE + regularization",
    )
    np.savez_compressed(
        config.artifact("reconstruction.npz", folder),
        **jax.device_get(params),
        **background_arrays,
        run_id=config.run_id,
        tag=config.tag,
        started_utc=config.started_utc,
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
        objective_history=history,
        regularization_active_history=reg_active,
        projection_active_history=projection_active,
        optimization_steps_completed=np.asarray(fixed[STEP_KEY]),
        initial_data_loss=initial["loss"],
        final_data_loss=final["loss"],
        initial_regularization=initial_reg,
        final_regularization=final_reg,
        initial_configured_regularization=initial_configured_reg,
        final_configured_regularization=final_configured_reg,
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
        probe_support=np.empty((0, 0), dtype=np.float32)
        if support is None
        else np.asarray(support),
    )
    if support is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        artist = ax.imshow(np.asarray(support), vmin=0, vmax=1, cmap="gray")
        ax.set(
            title="Probe amplitude support at sample",
            xlabel="Column (pixels)",
            ylabel="Row (pixels)",
        )
        fig.colorbar(artist, ax=ax)
        base._save_figure(fig, config.artifact("probe_support.png", folder))
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    for ax, enabled, label in zip(
        axes,
        (reg_active, projection_active),
        ("Regularization", "Projection"),
        strict=True,
    ):
        ax.step(
            np.arange(config.n_steps + 1), np.r_[enabled, enabled[-1]], where="post"
        )
        ax.set(ylabel=label, yticks=[0, 1], ylim=(-0.1, 1.1))
    axes[-1].set_xlabel("Zero-based full-pass AD update")
    base._save_figure(fig, config.artifact("constraint_schedule.png", folder))
    base._save_json(config.artifact("metrics.json", folder), metrics)
    if config.diagnostics:
        diagnostics_start = time.perf_counter()
        metrics["diagnostics"] = nanomax_diagnostics.write_diagnostics(
            folder,
            half_range_m=config.half_range_m,
            z_samples=config.z_samples,
            padding_factor=config.padding_factor,
            filename_prefix=config.run_id + "_",
        )
        metrics["diagnostics_seconds"] = time.perf_counter() - diagnostics_start
        base._save_json(config.artifact("metrics.json", folder), metrics)
    print(
        f"Modes={n_modes}: final data MSE={final['loss']:.6g}; regularization={final_reg:.6g}; "
        f"objective={metrics['final_objective']:.6g}; optimization={elapsed:.1f}s; saved {folder}",
        flush=True,
    )
    return metrics


def save_run_report(data, config, metrics):
    """Save comparison figures and a report whose links use the run identity."""
    base._save_json(config.artifact("summary.json"), metrics)
    fig, ax = plt.subplots(figsize=(7, 4))
    for result in metrics:
        with np.load(
            config.artifact(
                "reconstruction.npz", config.run_dir / f"modes_{result['modes']}"
            )
        ) as saved:
            history = saved["objective_history"]
        ax.semilogy(
            np.arange(len(history) + 1),
            np.r_[history, result["final_objective"]],
            label=f"{result['modes']} mode(s)",
        )
    ax.set(
        xlabel="Epoch",
        ylabel="Amplitude MSE + regularization",
        title="NanoMAX AD objective",
    )
    ax.legend()
    base._save_figure(fig, config.artifact("comparison.png"))
    geometry = metrics[0]["geometry"]
    report = [
        "# NanoMAX constrained AD reconstruction",
        "",
        f"Tag: `{config.tag}`; reconstruction started (UTC): {config.started_utc}.",
        f"Input: `{config.datapath}`; {len(data['total_ints'])} retained frames; crop {config.crop_shape}.",
        f"Detector: `{config.detector_name}` via `/entry/measurement/{config.detector_name}/frames`; resolved detector source is recorded in metrics.",
        (
            f"Energy: {geometry['energy_eV']:.6f} eV; wavelength: {geometry['wavelength_m']:.12g} m; "
            f"detector distance: {geometry['sample_detector_distance_m']:.6f} m; "
            f"sample pitches (y,x): {data['reconstruction_pix_size'] * 1e9} nm."
        ),
        "",
        (
            f"Complex sample TV weight: {config.sample_tv_weight:g}; complex probe TV weight: {config.probe_tv_weight:g}; "
            f"p={config.tv_p:g}, q={config.tv_q:g}, epsilon={config.tv_epsilon:g}."
        ),
        (
            f"TV RMS normalization: {config.tv_normalize}. With normalization, each spatial array is divided by its RMS amplitude "
            "before complex spatial differences, so compensated amplitude rescaling does not reduce its penalty. "
            "Loss histories show the total objective before each update; "
            "the final marker evaluates the final parameters. The MSE columns below exclude regularization."
        ),
        "",
        "| Modes | AD updates | Initial data MSE | Final data MSE | Final regularization | Final objective | Optimization incl. compile (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in metrics:
        report.append(
            f"| {item['modes']} | {item['epochs']} | {item['initial_loss']:.6g} | "
            f"{item['final_loss']:.6g} | {item['final_regularization']:.6g} | "
            f"{item['final_objective']:.6g} | {item['optimization_seconds_including_compile']:.1f} |"
        )
    report += [
        "",
        f"![Preparation]({config.artifact('preparation.png').name})",
        "",
        f"![Objective comparison]({config.artifact('comparison.png').name})",
        "",
    ]
    for item in metrics:
        n = item["modes"]
        folder = config.run_dir / f"modes_{n}"
        report += [f"## {n} probe modes", ""]
        report += [
            (
                f"Regularization schedule (TV/background/custom): `{config.regularization_schedule or 'all'}` "
                f"({item['regularization_active_steps']} active updates); "
                f"projection schedule: `{config.projection_schedule or 'all'}` "
                f"({item['projection_active_steps']} active updates). Steps are zero-based; STOP is exclusive."
            ),
            (
                "Penalties act in the gradient calculation; projections act after the corresponding Adam update. "
                "Adam state and frame shuffling are retained across schedule changes. The final objective uses "
                "the last update's penalty setting; configured regularization before scheduling is recorded separately in metrics. "
                "Objective jumps may reflect a schedule switch."
            ),
            "",
            f"![Constraint schedule]({config.artifact('constraint_schedule.png', folder).relative_to(config.run_dir)})",
            "",
        ]
        if "diagnostics" in item:
            analysis = item["diagnostics"]
            drift_y, drift_x = (
                np.asarray(analysis["position_drift_slope_m_per_shot_yx"]) * 1e9
            )
            diagnostic_report = config.artifact(
                "results.md", folder / "diagnostics"
            ).relative_to(config.run_dir)
            report += [
                (
                    f"[Full beam/scan diagnostics]({diagnostic_report}): SVD and coherent modes, mode powers, "
                    "HSV beam maps, propagation, focus/astigmatism, acquisition-order scan corrections, modal populations and beam moments."
                ),
                "",
                (
                    f"Estimated focus z from sample: **{analysis['estimated_focus_z_m'] * 1e3:+.6f} mm** "
                    f"(downstream positive); focus-to-sample: {analysis['estimated_focus_to_sample_m'] * 1e3:+.6f} mm. "
                    f"Descriptive position trends (y,x): {drift_y:+.6f}, {drift_x:+.6f} nm per acquisition shot. "
                    "Shot index is not elapsed time; focus is estimated from the reconstructed beam."
                ),
                "",
            ]
        if item.get("background", {}).get("enabled"):
            bg = item["background"]
            report += [
                (
                    f"Additive background: shared {config.background_grid_shape} log-intensity grid with per-frame levels; "
                    f"learning starts at update {config.background_start_step}. "
                    f"Mean background fraction of measured/predicted photons: {100 * bg['mean_measured_fraction']:.6f}% / "
                    f"{100 * bg['mean_predicted_fraction']:.6f}%; maximum per-frame measured fraction: {100 * bg['max_frame_measured_fraction']:.6f}%. "
                    f"Measured-photon cap: {100 * config.background_max_fraction:g}%."
                ),
                (
                    f"Background fraction in the bright-pixel region: {100 * bg['bright_pixel_predicted_fraction']:.6f}%; "
                    f"dominates total/bright-region signal: **{bg['dominates_signal']}**. "
                    f"Fraction of valid pixels where background exceeds signal: {100 * bg['fraction_valid_pixels_background_above_signal']:.3f}%. "
                    "Local background dominance in dark detector pixels differs from dominance of useful diffraction photons."
                ),
                (
                    f"Background L1 / spatial-gradient / log-scale-variance weights: "
                    f"{config.background_l1_weight:g} / {config.background_smoothness_weight:g} / {config.background_scale_weight:g}. "
                    "These penalties follow the regularization schedule; positivity, photon budget and coarse shape apply at all updates. "
                    "Signal/background separation is model dependent; a small photon fraction alone does not establish uniqueness."
                ),
                "",
                f"![Signal and additive background]({config.artifact('background.png', folder).relative_to(config.run_dir)})",
                "",
                f"![Background frame scaling]({config.artifact('background_scaling.png', folder).relative_to(config.run_dir)})",
                "",
            ]
        support = item["probe_support"]
        if support is not None:
            report += [
                (
                    f"Probe support: unity to radius {support['radius_px']:g} px, cosine decay over "
                    f"{support['decay_px']:g} px, zero at radius {support['zero_from_radius_px']:g} px; "
                    f"center (y,x) {support['center_yx_px']}."
                ),
                (
                    "The same amplitude support multiplies every mode after scheduled Adam updates. "
                    "Repeated application attenuates the tapered region repeatedly."
                ),
                "",
            ]
        for name, label in (
            ("sample.png", "Sample"),
            ("probe_initial.png", "Initial probe"),
            ("probe_final.png", "Final probe"),
            ("diffraction.png", "Diffraction fit"),
            ("loss.png", "Objective history"),
            ("probe_support.png", "Probe support"),
        ):
            if name == "probe_support.png" and support is None:
                continue
            report += [
                f"![{label}]({config.artifact(name, folder).relative_to(config.run_dir)})",
                "",
            ]
        report += [
            (
                f"Arrays: [reconstruction]({config.artifact('reconstruction.npz', folder).relative_to(config.run_dir)}); "
                f"[metrics]({config.artifact('metrics.json', folder).relative_to(config.run_dir)})."
            ),
            "",
        ]
    report += [
        (
            "Training-data fit only. Object/probe scale and phase ambiguities remain. "
            "The displayed sample is masked by approximate illumination coverage; defocus and detector geometry are supplied settings."
        ),
        "",
    ]
    config.artifact("results.md").write_text("\n".join(report))


def run_pipeline(
    config,
    modes=(1, 5),
    *,
    regularizer_fn=None,
    projection_fn=None,
    settings_source=None,
):
    """Load once and reconstruct each mode count into a fresh tagged run folder."""
    modes = tuple(modes)
    if (
        not modes
        or len(set(modes)) != len(modes)
        or any(n < 1 or int(n) != n for n in modes)
    ):
        raise ValueError("modes must contain distinct positive integers.")
    config.run_dir.mkdir(parents=True, exist_ok=False)
    config_data = asdict(config)
    config_data.update(
        run_id=config.run_id,
        run_dir=config.run_dir,
        modes=modes,
        custom_regularizer=_callback_name(regularizer_fn),
        custom_projection=_callback_name(projection_fn),
    )
    if settings_source is not None:
        snapshot = config.artifact("input_settings.json")
        snapshot.write_bytes(settings_source["content"])
        config_data["settings_source"] = {
            key: value for key, value in settings_source.items() if key != "content"
        }
        config_data["settings_source"]["snapshot"] = snapshot.name
    source_root = Path(__file__).parent
    sources = (
        Path(__file__).name,
        "pipeline_nanomax_upd.py",
        "ptycho_jax/data_loading/nanomax_dataloader.py",
        "ptycho_jax/transmission/loss_and_reg.py",
        "ptycho_jax/transmission/projectors.py",
        "ptycho_jax/transmission/optim_loop.py",
        "ptycho_jax/transmission/optimizers.py",
        "ptycho_jax/transmission/transmission_ptycho_models.py",
        "ptycho_jax/transmission/probe_initialization.py",
        "ptycho_jax/transmission/schedules.py",
        "ptycho_jax/transmission/settings.py",
        "ptycho_jax/transmission/background.py",
        "ptycho_jax/visualisation/probe_plotting.py",
        "ptycho_jax/visualisation/background_diagnostics.py",
        "ptycho_jax/visualisation/nanomax_diagnostics.py",
    )
    source_hashes = {}
    for name in sources:
        source = (source_root / name).read_bytes()
        source_hashes[name] = hashlib.sha256(source).hexdigest()
        snapshot_folder = config.run_dir / "source_snapshot" / Path(name).parent
        snapshot_folder.mkdir(parents=True, exist_ok=True)
        config.artifact(Path(name).name, snapshot_folder).write_bytes(source)
    config_data["source_sha256"] = source_hashes
    base._save_json(config.artifact("configuration.json"), config_data)
    print(
        f"Run {config.run_id}: loading {config.datapath}; saving {config.run_dir}",
        flush=True,
    )
    data = base.load_reconstruction_data(config)
    base.save_preparation_figure(
        data, config.run_dir, filename_prefix=config.run_id + "_"
    )
    metrics = [
        run_reconstruction(
            data, config, n, regularizer_fn=regularizer_fn, projection_fn=projection_fn
        )
        for n in modes
    ]
    save_run_report(data, config, metrics)
    print(f"Report: {config.artifact('results.md')}", flush=True)
    return metrics


def parse_configuration(argv=None):
    """Load JSON defaults, apply explicit CLI overrides, and validate before I/O."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON settings file; explicit CLI arguments override its values.",
    )
    parser.add_argument("--datapath", type=Path, default=base.DEFAULT_DATAPATH)
    parser.add_argument(
        "--detector-name",
        default="eiger4m",
        help="Scan HDF5 detector group; external frame-file links are followed automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nanomax"),
        help="Parent for a fresh scan_TIMESTAMP_TAG run folder.",
    )
    parser.add_argument(
        "--tag",
        help="Run label; default: five random ASCII letters. Spaces become underscores in filenames.",
    )
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 5])
    parser.add_argument(
        "--steps",
        dest="n_steps",
        type=int,
        default=400,
        help="Full-pass AD updates (epochs).",
    )
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
    parser.add_argument(
        "--sample-tv-weight",
        type=float,
        default=0.0,
        help="Nonnegative weight on complex sample TV; zero disables.",
    )
    parser.add_argument(
        "--probe-tv-weight",
        type=float,
        default=0.0,
        help="Nonnegative weight on complex probe-mode TV; zero disables.",
    )
    parser.add_argument("--tv-p", type=float, default=2.0)
    parser.add_argument("--tv-q", type=float, default=1.0)
    parser.add_argument("--tv-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--tv-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each sample/mode by spatial RMS before TV (default). --no-tv-normalize selects raw TV.",
    )
    parser.add_argument(
        "--regularization-schedule",
        nargs="+",
        metavar="RANGE",
        help="TV/custom-penalty updates: INDEX or START:STOP[:EVERY], multiple ranges allowed; default all. Use 'none' to disable.",
    )
    parser.add_argument(
        "--projection-schedule",
        nargs="+",
        metavar="RANGE",
        help="Post-update constraint schedule, independently of TV; same syntax and defaults.",
    )
    parser.add_argument(
        "--probe-support",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Multiply all probe modes by a smooth support after scheduled Adam updates.",
    )
    parser.add_argument(
        "--background",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize one additive detector background with per-frame scaling (default disabled).",
    )
    parser.add_argument(
        "--background-grid-shape",
        type=int,
        nargs=2,
        default=[16, 16],
        help="Coarse shared background logits [rows columns], interpolated in centered detector coordinates.",
    )
    parser.add_argument(
        "--background-start-step",
        type=int,
        default=100,
        help="Zero-based update at which background parameter learning begins; its initial background is always modeled.",
    )
    for name, default, description in (
        (
            "initial-fraction",
            1e-4,
            "Initial background fraction of measured valid photons.",
        ),
        (
            "max-fraction",
            0.05,
            "Hard upper bound per frame on background/measured valid photons.",
        ),
        ("learning-rate", 0.03, "Adam rate for the shared background shape logits."),
        (
            "scale-learning-rate",
            0.03,
            "Adam rate for the per-frame background level logits.",
        ),
        (
            "l1-weight",
            0.01,
            "Penalty on mean background counts per valid detector pixel.",
        ),
        (
            "smoothness-weight",
            1.0,
            "Penalty on squared spatial gradients of the normalized shared background.",
        ),
        (
            "scale-weight",
            0.1,
            "Penalty on variance of log per-frame background photon counts.",
        ),
    ):
        parser.add_argument(
            "--background-" + name, type=float, default=default, help=description
        )
    parser.add_argument(
        "--probe-support-radius-px",
        type=float,
        help="Flat support radius in sample-plane pixels; default 0.4 * min(crop_shape).",
    )
    parser.add_argument(
        "--probe-support-decay-px",
        type=float,
        help="Cosine decay width in sample-plane pixels; default 0.1 * min(crop_shape).",
    )
    parser.add_argument(
        "--probe-support-center-px",
        type=float,
        nargs=2,
        metavar=("ROW", "COL"),
        help="Support center; default geometric probe center.",
    )
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run orthogonalization, propagation, focus and drift diagnostics after fitting (default enabled).",
    )
    parser.add_argument(
        "--half-range-m",
        type=float,
        help="Propagation sweep ±z about sample; default twice abs(defocus), i.e. ±3 mm for this scan.",
    )
    parser.add_argument(
        "--z-samples",
        type=int,
        default=201,
        help="Odd number >=5 of propagation planes.",
    )
    parser.add_argument(
        "--padding-factor",
        type=int,
        default=3,
        help="Transverse propagation zero-padding factor.",
    )
    for parameter, default in (
        ("sample", 0.03),
        ("probe", 0.3),
        ("weights", 0.003),
        ("positions", 0.003),
    ):
        parser.add_argument(f"--{parameter}-learning-rate", type=float, default=default)
    initial = parser.parse_args(argv)
    settings_source = None
    if initial.config is not None:
        try:
            settings, settings_source = load_json_settings(
                initial.config, ReconstructionConfig
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        parser.set_defaults(**settings)
    args = vars(parser.parse_args(argv))
    args.pop("config")
    try:
        validate_setting_values(args)
    except ValueError as error:
        parser.error(str(error))
    device_name = args.pop("device")
    modes = args.pop("modes")
    if args["tag"] is None:
        args.pop("tag")
    for key in (
        "crop_shape",
        "crop_center",
        "probe_support_center_px",
        "regularization_schedule",
        "projection_schedule",
        "intensity_limits",
        "background_grid_shape",
    ):
        if args.get(key) is not None:
            args[key] = tuple(args[key])
    try:
        config = ReconstructionConfig(**args)
        if len(set(modes)) != len(modes) or min(modes) < 1:
            raise ValueError("--modes must contain distinct positive integers.")
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if settings_source is not None:
        settings_source.update(
            requested_device=device_name,
            arguments=argv,
            working_directory=str(Path.cwd()),
        )
    return config, modes, device_name, settings_source


def main(argv=None):
    config, modes, device_name, settings_source = parse_configuration(argv)
    device = jax.devices(device_name)[0] if device_name else jax.devices()[0]
    with jax.default_device(device):
        run_pipeline(config, modes, settings_source=settings_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
