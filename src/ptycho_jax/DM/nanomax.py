"""NanoMAX two-mode DM initialization followed by fixed-position AD refinement."""

from __future__ import annotations
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse, json, time, hashlib
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pipeline_nanomax_upd as base
from .config import DMConfig
from .geometry import make_geometry
from .init import initialize_state, make_data
from .dm import run_dm
from .metrics import amplitude_metrics
from .handoff import to_ad_parameters, evaluate_ad, refine_ad
from .plotting import save_snapshot, save_convergence


def save_json(path, data):
    def default(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    Path(path).write_text(
        json.dumps(data, default=default, indent=2, allow_nan=False) + "\n"
    )


def make_start(loaded, base_cfg, guess_defocus):
    """Prepare two orthonormal modes with a specified probe-guess defocus only."""
    case_data = {
        **loaded,
        "experimental_parameters": {
            **loaded["experimental_parameters"],
            "assumed_defocus_at_sample_m": guess_defocus,
        },
    }
    ref = base.initialize_reconstruction(
        case_data, replace(base_cfg, defocus_m=guess_defocus), n_modes=1
    )
    probe0 = np.asarray(ref["params"]["probe_modes"][0], dtype=np.complex64)
    geom = make_geometry(
        np.asarray(ref["fixed"]["sample_positions"]),
        np.asarray(loaded["float_parts"], dtype=np.float32),
        ref["params"]["sample"].shape,
        loaded["mean_probe"].shape,
        base_cfg.shift_margin,
    )
    data = make_data(
        loaded["measured_intensities"], loaded["detector_mask"], dtype=jnp.float32
    )
    state = initialize_state(probe0, 2, data, geom, seed=base_cfg.seed)
    return state, data, geom, ref["object_pixel_origin"]


def run_case(loaded, base_cfg, dm_cfg, output, *, factor, dm_iterations, ad_steps):
    name = "correct_defocus" if factor == 1 else f"defocus_{factor:.2f}x"
    folder = Path(output) / name
    folder.mkdir(parents=True)
    nominal = base_cfg.defocus_m
    guess = nominal * factor
    print(
        f"CASE {name}: nominal={nominal * 1e3:.4f}mm; probe guess={guess * 1e3:.4f}mm",
        flush=True,
    )
    state, data, geom, origin = make_start(loaded, base_cfg, guess)
    initial_metric = amplitude_metrics(state.psi, data)
    naive, fixed = to_ad_parameters(state, geom)
    initial_ad = evaluate_ad(naive, fixed, geom, data, base_cfg.batch_size)
    save_snapshot(
        folder / "initial",
        naive,
        fixed,
        loaded["reconstruction_pix_size"],
        origin,
        title=f"Initial guess: defocus {guess * 1e3:.3f} mm",
        frame_indices=loaded["frame_indices"],
        margin=geom.margin,
    )
    trace = []

    def snapshot(current, block):
        if int(current.it) % 50 == 0 or int(current.it) == dm_iterations:
            trace.append(
                {
                    "iteration": int(current.it),
                    "model_mse": float(block["model_mse"][-1]),
                }
            )
            np.savez_compressed(
                folder / f"dm_checkpoint_{int(current.it):04d}.npz",
                sample=np.asarray(current.obj),
                unit_probe=np.asarray(current.probe),
                a=np.asarray(current.a),
            )

    start = time.perf_counter()
    state, history = run_dm(
        state,
        data,
        geom,
        dm_cfg,
        dm_iterations,
        log_every=10,
        callback=snapshot,
        debug_dir=folder / "debug",
    )
    elapsed = time.perf_counter() - start
    dm_params, dm_fixed = to_ad_parameters(state, geom)
    handoff = evaluate_ad(dm_params, dm_fixed, geom, data, base_cfg.batch_size)
    dm_mse = float(history["model_mse"][-1])
    relative = abs(handoff["loss"] - dm_mse) / max(dm_mse, 1e-30)
    if relative > 1e-4 and abs(handoff["loss"] - dm_mse) > 1e-5:
        raise RuntimeError(f"DM/AD handoff mismatch: DM={dm_mse}, AD={handoff['loss']}")
    save_snapshot(
        folder / "dm",
        dm_params,
        dm_fixed,
        loaded["reconstruction_pix_size"],
        origin,
        title=f"After {dm_iterations} DM iterations",
        frame_indices=loaded["frame_indices"],
        margin=geom.margin,
    )
    np.savez_compressed(folder / "dm_history.npz", **history)
    np.savez_compressed(
        folder / "dm_factors.npz",
        unit_probe=np.asarray(state.probe),
        a=np.asarray(state.a),
        sample=np.asarray(state.obj),
        n=np.asarray(geom.n),
        delta=np.asarray(geom.delta),
    )
    print(
        f"{name}: DM final={dm_mse:.6g}, AD handoff={handoff['loss']:.6g}, relative mismatch={relative:.3g}",
        flush=True,
    )
    del state
    refined, ad_history, warm_metrics = refine_ad(
        dm_params,
        dm_fixed,
        geom,
        data,
        n_steps=ad_steps,
        batch_size=base_cfg.batch_size,
        seed=base_cfg.seed,
    )
    save_snapshot(
        folder / "ad",
        refined,
        dm_fixed,
        loaded["reconstruction_pix_size"],
        origin,
        title=f"DM → {ad_steps} AD epochs",
        frame_indices=loaded["frame_indices"],
        margin=geom.margin,
    )
    cold, cold_history, cold_metrics = refine_ad(
        naive,
        fixed,
        geom,
        data,
        n_steps=ad_steps,
        batch_size=base_cfg.batch_size,
        seed=base_cfg.seed,
    )
    save_snapshot(
        folder / "ad_only",
        cold,
        fixed,
        loaded["reconstruction_pix_size"],
        origin,
        title=f"AD only: {ad_steps} epochs",
        frame_indices=loaded["frame_indices"],
        margin=geom.margin,
    )
    final = evaluate_ad(refined, dm_fixed, geom, data, base_cfg.batch_size)
    result = {
        "case": name,
        "nominal_defocus_m": nominal,
        "initialization_defocus_m": guess,
        "defocus_factor": factor,
        "positions": len(geom.n),
        "probe_modes": 2,
        "positions_frozen": True,
        "dm_iterations": dm_iterations,
        "ad_epochs": ad_steps,
        "initial_mse": float(initial_metric["model_mse"]),
        "initial_reference_mse": initial_ad["loss"],
        "dm_mse": dm_mse,
        "handoff_mse": handoff["loss"],
        "handoff_relative_difference": relative,
        "dm_seconds_including_compile": elapsed,
        "dm_ad": warm_metrics,
        "ad_only": cold_metrics,
        "geometry": loaded["experimental_parameters"],
        "sample_pixel_size_m": loaded["reconstruction_pix_size"],
        "device": str(refined["sample"].device),
        "device_kind": refined["sample"].device.device_kind,
        "jax_version": jax.__version__,
        "dm_checkpoints": trace,
        "ad_weight_family": "full complex 2x2 matrices; DM starts on nonnegative diagonal",
        "dm_weight_family": "nonnegative amplitude a[shot,mode]; intensity weights a**2",
    }
    np.savez_compressed(
        folder / "ad_history.npz", warm=ad_history, ad_only=cold_history
    )
    save_convergence(folder, history, ad_history, cold_history, result)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, image, label in (
        (axes[0], loaded["measured_intensities"][0], "Measured"),
        (axes[1], handoff["first_prediction"], "After DM"),
        (axes[2], final["first_prediction"], "After DM → AD"),
    ):
        value = np.where(loaded["detector_mask"], np.log10(1 + image), np.nan)
        ax.imshow(np.fft.fftshift(value), cmap="magma")
        ax.set_title(label + " [log10(1+I)]")
    fig.savefig(folder / "diffraction_comparison.png", dpi=150)
    plt.close(fig)
    save_json(folder / "metrics.json", result)
    print(
        f"{name}: DM {dm_mse:.6g} → AD {warm_metrics['final_loss']:.6g}; "
        f"AD-only {cold_metrics['final_loss']:.6g}",
        flush=True,
    )
    return result


def write_report(output, results):
    lines = [
        "# NanoMAX: difference-map initialization and AD continuation",
        "",
        "Two beam modes and shot-specific weights. Integer and fractional scan positions are fixed in both stages. "
        "The incorrect-defocus case changes only the probe initialization defocus, not the wavelength, detector distance, measured data or scan positions.",
        "",
        "| Probe guess | Initial MSE | After DM | AD handoff | DM → AD | AD only | DM / AD iterations |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['initialization_defocus_m'] * 1e3:.3f} mm | {r['initial_mse']:.6g} | "
            f"{r['dm_mse']:.6g} | {r['handoff_mse']:.6g} | {r['dm_ad']['final_loss']:.6g} | "
            f"{r['ad_only']['final_loss']:.6g} | {r['dm_iterations']} / {r['ad_epochs']} |"
        )
    for r in results:
        case = r["case"]
        lines += [
            "",
            f"## {case}",
            "",
            f"![Convergence]({case}/convergence.png)",
            "",
            f"![DM sample]({case}/dm/sample.png)",
            "",
            f"![DM-to-AD sample]({case}/ad/sample.png)",
            "",
            f"![AD-only sample]({case}/ad_only/sample.png)",
            "",
            f"![DM modes]({case}/dm/probe.png)",
            "",
            f"![Refined modes]({case}/ad/probe.png)",
            "",
            f"![Shot weights]({case}/dm/weights.png)",
            "",
            f"![Diffraction fit]({case}/diffraction_comparison.png)",
            "",
            f"[Numeric metrics]({case}/metrics.json).",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "DM uses the exact reference tapered-object selector and orthonormal FFT, analytical nonnegative weight "
        "and probe updates, and a matrix-free damped CG object solve. There is no gradient descent inside DM. "
        "The transferred solution is the last overlap model, not the raw exit-wave iterate. DM Fourier/update "
        "residuals need not decrease monotonically. The AD loss is evaluated with the untouched reference model; "
        "positions are fixed parameters and absent from the optimizer. Full float32 contraction precision is "
        "requested in the new DM/reference wrappers to avoid default GPU tensor-math approximation.",
        "",
        "AD starts with the exact DM field, then permits full complex shot-specific 2×2 modal matrices, "
        "as in the previous AD model. The additional mixing freedom can also lower loss. AD-only controls use "
        "the same initial DM parameterization and the same AD learning rates, fixed geometry and AD epoch count. "
        "DM→AD has additional DM computation, so this is an initialization comparison rather than an equal-total-compute benchmark.",
        "",
        "The 1.50 mm setting is the supplied nominal defocus, not an independently calibrated true value. "
        "No empty-beam measurement or held-out data is used. Object/probe gauges, weak modes and fixed-position "
        "measurement errors limit interpretation. The 25.77 nm sample spacing is not a measured resolution.",
        "",
    ]
    (Path(output) / "results.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", type=Path, default=base.DEFAULT_DATAPATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--crop-shape", nargs=2, type=int, default=[512, 512])
    parser.add_argument("--crop-center", nargs=2, type=float)
    parser.add_argument("--dm-iterations", type=int, default=200)
    parser.add_argument("--ad-steps", type=int, default=400)
    parser.add_argument("--dm-inner", type=int, default=3)
    parser.add_argument("--cg-iterations", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=13)
    parser.add_argument("--defocus-m", type=float, default=0.0015)
    parser.add_argument("--defocus-factors", nargs="+", type=float, default=[1.0, 1.3])
    parser.add_argument("--detector-distance-m", type=float, default=7.180)
    parser.add_argument("--detector-pixel-size-m", type=float, default=75e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "gpu"])
    args = parser.parse_args(argv)
    if min(args.dm_iterations, args.ad_steps, args.batch_size) < 1:
        parser.error("Iteration counts and batch size must be positive.")
    if len(set(args.defocus_factors)) != len(args.defocus_factors):
        parser.error("Defocus factors must be distinct.")
    output = args.output_dir or Path("results/nanomax_dm_ad") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    if output.exists():
        parser.error("Choose a fresh output directory: " + str(output))
    output.mkdir(parents=True)
    cfg = base.ReconstructionConfig(
        datapath=args.datapath,
        output_dir=output,
        crop_shape=tuple(args.crop_shape),
        crop_center=None if args.crop_center is None else tuple(args.crop_center),
        detector_distance_m=args.detector_distance_m,
        detector_pixel_size_m=args.detector_pixel_size_m,
        defocus_m=args.defocus_m,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    dm_cfg = DMConfig(
        n_inner=args.dm_inner, object_cg_iterations=args.cg_iterations, beta=args.beta
    )
    device = jax.devices(args.device)[0] if args.device else jax.devices()[0]
    with jax.default_device(device):
        loaded = base.load_reconstruction_data(cfg)
        base.save_preparation_figure(loaded, output)
        save_json(
            output / "configuration.json",
            {
                "reconstruction": asdict(cfg),
                "DM": asdict(dm_cfg),
                "defocus_factors": args.defocus_factors,
                "DM_iterations": args.dm_iterations,
                "AD_steps": args.ad_steps,
                "contraction_precision": "highest",
                "production_array_dtype": "float32/complex64",
                "x64_enabled": jax.config.jax_enable_x64,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "sources": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        *Path(__file__).parent.glob("*.py"),
                        Path(base.__file__),
                        Path(__file__).parents[1]
                        / "transmission/transmission_ptycho_models.py",
                    ]
                },
            },
        )
        results = []
        for factor in args.defocus_factors:
            results.append(
                run_case(
                    loaded,
                    cfg,
                    dm_cfg,
                    output,
                    factor=factor,
                    dm_iterations=args.dm_iterations,
                    ad_steps=args.ad_steps,
                )
            )
            save_json(output / "summary.json", results)
            write_report(output, results)
    print("Results:", output / "results.md", flush=True)
    return 0
