"""Run the NanoMAX reconstruction, then save separate beam diagnostics.

Defaults to five modes. Standard reconstruction options are forwarded unchanged
to pipeline_nanomax_upd.py. Old pipelines and their saved runs are untouched.

Example:
    .venv/bin/python src/pipeline_nanomax_diagnostics.py \\
      --output-dir results/nanomax/scan_000082_diagnostics --steps 400 --device gpu
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pipeline_nanomax_upd as reconstruction
from ptycho_jax.visualisation import nanomax_diagnostics as diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All other options (e.g. --datapath, --steps, --batch-size, --crop-shape, "
               "--defocus-m, --device) are forwarded to pipeline_nanomax_upd.py.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--modes", type=int, nargs="+", default=[5])
    parser.add_argument("--half-range-m", type=float, help="Propagation ±z about sample; default 2*abs(defocus).")
    parser.add_argument("--z-samples", type=int, default=201, help="Odd number of propagation planes.")
    parser.add_argument("--padding-factor", type=int, default=3, help="Transverse zero-padding factor.")
    args, forwarded = parser.parse_known_args(argv)
    if args.z_samples < 5 or args.z_samples % 2 != 1:
        parser.error("--z-samples must be an odd integer >= 5.")
    if args.padding_factor < 1 or (args.half_range_m is not None and args.half_range_m <= 0):
        parser.error("Padding and any explicit propagation half-range must be positive.")
    output = args.output_dir or Path("results/nanomax") / (
        "diagnostics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    reconstruction.main([
        *forwarded, "--output-dir", str(output), "--modes", *map(str, args.modes),
    ])
    return analyze_run(output, args.modes, half_range_m=args.half_range_m,
                       z_samples=args.z_samples, padding_factor=args.padding_factor)


def analyze_run(output, mode_counts, *, half_range_m=None, z_samples=201, padding_factor=3):
    """Append or regenerate diagnostics for completed runs without refitting."""
    output = Path(output)
    summaries = {}
    for mode_count in mode_counts:
        summaries[str(mode_count)] = diagnostics.write_diagnostics(
            output / f"modes_{mode_count}", half_range_m=half_range_m,
            z_samples=z_samples, padding_factor=padding_factor,
        )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "modes": mode_counts, "half_range_m": half_range_m,
        "z_samples": z_samples, "padding_factor": padding_factor,
        "half_range_default_interpretation": "twice the supplied focus-to-sample defocus, about sample",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(diagnostics.__file__), Path(reconstruction.__file__))
        },
    }
    for name, value in (("diagnostics_configuration.json", provenance), ("diagnostics_summary.json", summaries)):
        (output / name).write_text(json.dumps(value, default=diagnostics._json_default,
                                             indent=2, allow_nan=False) + "\n")
    report = [
        "# NanoMAX reconstruction with beam diagnostics", "",
        "[Reconstruction report and figures](results.md).", "",
        "| Modes | Estimated focus z from sample (mm) | Focus-to-sample (mm) | x/y waist z (mm) |",
        "| --- | --- | --- | --- |",
    ]
    for mode_count, result in summaries.items():
        report.append(f"| {mode_count} | {result['estimated_focus_z_m']*1e3:+.6f} | "
                      f"{result['estimated_focus_to_sample_m']*1e3:+.6f} | "
                      f"{result['focus_x_z_m']*1e3:+.6f} / {result['focus_y_z_m']*1e3:+.6f} |")
    for mode_count in summaries:
        report += ["", f"[All {mode_count}-mode diagnostics and separate images]"
                   f"(modes_{mode_count}/diagnostics/results.md).", ""]
    report += [
        "Positive propagation z is downstream from the sample. Focus is estimated from the reconstructed "
        "wavefront; it is not an independently calibrated true focus. The propagation range defaults "
        "to ±2 times the sample defocus (±3 mm for this scan), rather than ±2 objective focal lengths.", "",
        "Evolution plots restore original acquisition frame order before displaying mode weights, beam "
        "moments or position corrections. Shot index is not elapsed time.", "",
        "[Numeric summary](diagnostics_summary.json); [analysis provenance](diagnostics_configuration.json).", "",
    ]
    (output / "diagnostics_results.md").write_text("\n".join(report))
    print(f"Full diagnostics report: {output / 'diagnostics_results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
