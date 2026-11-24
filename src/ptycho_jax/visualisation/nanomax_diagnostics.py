"""Physical beam, focus and acquisition-order diagnostics for NanoMAX results.

The fitted field is E[p,m] = sum_n P[n] W[p,n,m]. Basis transformations must
therefore transform W too. No dense spatial mutual-intensity matrix is built.
Positive Fresnel z means downstream from the reconstructed sample plane.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from scipy import fft, linalg


PROPAGATION_SOURCE = "https://qiweb.tudelft.nl/aoi/wavefieldpropagation/wavefieldpropagation/"
BEAM_WIDTH_SOURCE = "https://www.nist.gov/sites/default/files/documents/calibrations/SP250-87.pdf"


def _anchor_phases(modes, weights):
    flat = modes.reshape(len(modes), -1)
    anchors = flat[np.arange(len(flat)), np.argmax(abs(flat), axis=1)]
    phase = np.exp(-1j * np.angle(anchors))
    return modes * phase[:, None, None], weights / phase[None, :, None]


def orthogonalize_probe_ensemble(probe_modes, modal_weights) -> dict:
    """Return an SVD basis and shot-averaged coherent modes, preserving fields.

    P = U S Vh, Q = Vh, C = S U.T W (transpose, without conjugation).
    Diagonalizing mean(C C†) then gives coherent modes and physical mean powers.
    """
    modes = np.asarray(probe_modes, dtype=np.complex128)
    weights = np.asarray(modal_weights, dtype=np.complex128)
    if modes.ndim != 3 or weights.ndim != 3 or weights.shape[1] != len(modes):
        raise ValueError("Expected modes (N,Y,X) and modal_weights (shots,N,M).")
    if min(*modes.shape, *weights.shape) < 1 or not np.isfinite(modes).all() or not np.isfinite(weights).all():
        raise ValueError("Modes and weights must be nonempty and finite.")
    u, singular, vh = linalg.svd(modes.reshape(len(modes), -1), full_matrices=False,
                                 check_finite=False, lapack_driver="gesvd")
    q = vh.reshape((-1, *modes.shape[-2:]))
    coefficients = np.einsum("nk,pnm->pkm", u, weights) * singular[None, :, None]
    q, coefficients = _anchor_phases(q, coefficients)
    covariance = np.einsum("pnm,pkm->nk", coefficients, coefficients.conj()) / len(weights)
    powers, vectors = linalg.eigh(covariance)
    powers, vectors = np.maximum(powers[::-1], 0), vectors[:, ::-1]
    if powers.sum() <= 0:
        raise ValueError("Reconstructed beam ensemble has zero power.")
    coherent = (vectors.T @ q.reshape(len(q), -1)).reshape(q.shape)
    coherent_weights = np.einsum("nk,pnm->pkm", vectors.conj(), coefficients)
    coherent, coherent_weights = _anchor_phases(coherent, coherent_weights)
    return {
        "svd_modes": q, "svd_weights": coefficients, "singular_values": singular,
        "coherent_modes": coherent, "coherent_weights": coherent_weights,
        "mode_powers": powers, "mode_fractions": powers / powers.sum(),
    }


def complex_beam_rgb(field, *, amplitude_scale=None, gamma=0.5):
    """HSV: phase sets hue, amplitude sets brightness, and zero is black."""
    field = np.asarray(field)
    amplitude = abs(field)
    scale = np.max(amplitude) if amplitude_scale is None else amplitude_scale
    if scale <= 0:
        scale = 1.0
    hue = (np.angle(field) + np.pi) / (2 * np.pi)
    value = np.clip(amplitude / scale, 0, 1) ** gamma
    return hsv_to_rgb(np.stack((hue, np.ones_like(hue), value), axis=-1))


def _frequency_squared(shape, pixel_size_m):
    dy, dx = np.broadcast_to(np.asarray(pixel_size_m, dtype=float), (2,))
    if min(dy, dx) <= 0 or not np.isfinite([dy, dx]).all():
        raise ValueError("Sample pixel sizes must be finite and positive.")
    return fft.fftfreq(shape[-2], dy)[:, None] ** 2 + fft.fftfreq(shape[-1], dx)[None, :] ** 2


def fresnel_propagate(field, pixel_size_m, wavelength_m, z_m):
    """Unitary paraxial angular-spectrum propagation; global carrier is omitted."""
    if wavelength_m <= 0 or not np.isfinite([wavelength_m, z_m]).all():
        raise ValueError("Wavelength must be positive and propagation distance finite.")
    field = np.asarray(field)
    transfer = np.exp(-1j * np.pi * wavelength_m * z_m * _frequency_squared(field.shape, pixel_size_m))
    return fft.ifft2(fft.fft2(field, norm="ortho", workers=2) * transfer, norm="ortho", workers=2)


def _interpolated_minimum(z, values):
    index = int(np.argmin(values))
    boundary = index in (0, len(z) - 1)
    estimate = float(z[index])
    if not boundary:
        step = z[index + 1] - z[index]
        a, b, _ = np.polyfit([-1, 0, 1], values[index - 1:index + 2], 2)
        if a > 0:
            offset = -b / (2 * a)
            if abs(offset) <= 1:
                estimate += float(step * offset)
    return estimate, boundary


def analyze_propagation(modes, powers, pixel_size_m, wavelength_m, *,
                        half_range_m, z_samples=201, padding_factor=3) -> dict:
    """Propagate the mean incoherent beam and estimate second-moment waists.

    Stream z planes; retain projections and selected intensity planes rather
    than a complete 3D complex volume. Zero padding reduces periodic wraparound.
    Focus minimizes sigma_x²+sigma_y²; x/y waists and an effective-area minimum
    are also reported to expose astigmatism or sensitivity to broad tails.
    """
    modes = np.asarray(modes, dtype=np.complex128)
    powers = np.asarray(powers, dtype=float)
    if modes.ndim != 3 or powers.shape != (len(modes),) or np.any(powers < 0) or powers.sum() <= 0:
        raise ValueError("Expected nonnegative modal powers and nonempty modes.")
    if half_range_m <= 0 or not np.isfinite(half_range_m):
        raise ValueError("half_range_m must be finite and positive.")
    if not isinstance(z_samples, int) or z_samples < 5 or z_samples % 2 != 1:
        raise ValueError("z_samples must be an odd integer >= 5, including z=0.")
    if not isinstance(padding_factor, int) or padding_factor < 1:
        raise ValueError("padding_factor must be a positive integer.")
    if wavelength_m <= 0 or not np.isfinite(wavelength_m):
        raise ValueError("wavelength_m must be finite and positive.")
    dy, dx = np.broadcast_to(np.asarray(pixel_size_m, dtype=float), (2,))
    ny, nx = modes.shape[-2:]
    py, px = ny * (padding_factor - 1), nx * (padding_factor - 1)
    padded = np.pad(modes * np.sqrt(powers)[:, None, None],
                    ((0, 0), (py // 2, py - py // 2), (px // 2, px - px // 2)))
    y = (np.arange(padded.shape[-2]) - ny // 2 - py // 2) * dy
    x = (np.arange(padded.shape[-1]) - nx // 2 - px // 2) * dx
    frequency_squared = _frequency_squared(padded.shape, (dy, dx))
    spectrum = fft.fft2(padded, norm="ortho", workers=2)
    expected_energy = np.sum(abs(padded) ** 2)
    z = np.linspace(-half_range_m, half_range_m, z_samples)
    projections_x, projections_y, variance_x, variance_y = [], [], [], []
    centroids, energies, effective_area, peak, border_fraction = [], [], [], [], []
    selected_indices = set(np.linspace(0, z_samples - 1, 7).astype(int))
    selected_planes, selected_z = [], []
    by, bx = max(1, len(y) // 20), max(1, len(x) // 20)

    def intensity_at(distance):
        transfer = np.exp(-1j * np.pi * wavelength_m * distance * frequency_squared)
        field = fft.ifft2(spectrum * transfer, norm="ortho", workers=2)
        return np.sum(field.real ** 2 + field.imag ** 2, axis=0)

    for index, distance in enumerate(z):
        intensity = intensity_at(distance)
        marginal_x, marginal_y = intensity.sum(axis=0), intensity.sum(axis=1)
        energy = marginal_x.sum()
        mx, my = np.dot(marginal_x, x) / energy, np.dot(marginal_y, y) / energy
        variance_x.append(max(0.0, np.dot(marginal_x, (x - mx) ** 2) / energy))
        variance_y.append(max(0.0, np.dot(marginal_y, (y - my) ** 2) / energy))
        projections_x.append(marginal_x)
        projections_y.append(marginal_y)
        centroids.append([my, mx])
        energies.append(energy)
        effective_area.append(energy ** 2 / np.sum(intensity ** 2) * dy * dx)
        peak.append(intensity.max() / (dy * dx))
        border_fraction.append(1 - intensity[by:-by, bx:-bx].sum() / energy)
        if index in selected_indices:
            selected_planes.append(intensity.astype(np.float32))
            selected_z.append(distance)
    vx, vy = np.asarray(variance_x), np.asarray(variance_y)
    focus_z, at_boundary = _interpolated_minimum(z, vx + vy)
    focus_x_z, boundary_x = _interpolated_minimum(z, vx)
    focus_y_z, boundary_y = _interpolated_minimum(z, vy)
    area_focus_z, area_boundary = _interpolated_minimum(z, np.asarray(effective_area))
    focus_intensity = intensity_at(focus_z)
    focus_px, focus_py = focus_intensity.sum(axis=0), focus_intensity.sum(axis=1)
    focus_energy = focus_px.sum()
    focus_mx, focus_my = np.dot(focus_px, x) / focus_energy, np.dot(focus_py, y) / focus_energy
    focus_vx = np.dot(focus_px, (x-focus_mx)**2) / focus_energy
    focus_vy = np.dot(focus_py, (y-focus_my)**2) / focus_energy
    return {
        "z_m": z, "x_m": x, "y_m": y,
        "projection_x": np.asarray(projections_x), "projection_y": np.asarray(projections_y),
        "variance_x_m2": vx, "variance_y_m2": vy,
        "d4sigma_x_m": 4 * np.sqrt(vx), "d4sigma_y_m": 4 * np.sqrt(vy),
        "centroids_m": np.asarray(centroids), "energies": np.asarray(energies),
        "effective_area_m2": np.asarray(effective_area), "peak_intensity_per_m2": np.asarray(peak),
        "border_energy_fraction": np.asarray(border_fraction),
        "focus_z_m": focus_z, "focus_x_z_m": focus_x_z, "focus_y_z_m": focus_y_z,
        "focus_at_boundary": at_boundary, "focus_x_at_boundary": boundary_x,
        "focus_y_at_boundary": boundary_y, "effective_area_focus_z_m": area_focus_z,
        "effective_area_focus_at_boundary": area_boundary,
        "focus_d4sigma_x_m": float(4 * np.sqrt(focus_vx)),
        "focus_d4sigma_y_m": float(4 * np.sqrt(focus_vy)),
        "focus_plane": focus_intensity.astype(np.float32),
        "selected_planes": np.asarray(selected_planes), "selected_z_m": np.asarray(selected_z),
        "maximum_energy_relative_error": float(np.max(abs(np.asarray(energies) / expected_energy - 1))),
        "padding_factor": padding_factor,
    }


def analyze_scan_positions(initial_pix, corrections_pix, frame_indices, pixel_size_m) -> dict:
    """Restore acquisition order and fit a descriptive correction trend per shot."""
    initial = np.asarray(initial_pix, dtype=float)
    corrections = np.asarray(corrections_pix, dtype=float)
    ids = np.asarray(frame_indices, dtype=int)
    if initial.shape != corrections.shape or initial.shape != (len(ids), 2) or len(np.unique(ids)) != len(ids):
        raise ValueError("Expected aligned (shots,2) positions and unique original frame indices.")
    order = np.argsort(ids)
    ids = ids[order]
    pixel = np.broadcast_to(np.asarray(pixel_size_m), (2,))
    initial, corrections = initial[order] * pixel, corrections[order] * pixel
    if len(ids) > 1:
        design = np.column_stack((ids - ids.mean(), np.ones(len(ids))))
        coefficients = np.linalg.lstsq(design, corrections, rcond=None)[0]
        trend = design @ coefficients
        slope = coefficients[0]
    else:
        trend, slope = corrections.copy(), np.zeros(2)
    residual = corrections - trend
    variation = np.sum((corrections - corrections.mean(axis=0)) ** 2, axis=0)
    r_squared = np.zeros(2)
    np.divide(np.sum(residual ** 2, axis=0), variation, out=r_squared, where=variation > 0)
    r_squared = np.where(variation > 0, 1 - r_squared, 0)
    return {
        "order": order, "frame_indices": ids, "initial_m": initial,
        "corrected_m": initial + corrections, "corrections_m": corrections,
        "drift_trend_m": trend, "drift_slope_m_per_shot": slope,
        "drift_residual_rms_m": np.sqrt(np.mean(residual ** 2, axis=0)),
        "drift_r_squared": r_squared,
    }


def analyze_modal_evolution(decomposition, frame_indices, pixel_size_m) -> dict:
    """Beam powers and spatial moments in a fixed, physically orthogonal basis."""
    order = np.argsort(frame_indices)
    coefficients = decomposition["coherent_weights"][order]
    covariance = np.einsum("pnm,pkm->pnk", coefficients, coefficients.conj())
    mode_power = np.sum(abs(coefficients) ** 2, axis=-1)
    total = mode_power.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Every shot must have positive reconstructed beam power.")
    modes = decomposition["coherent_modes"]
    dy, dx = np.broadcast_to(np.asarray(pixel_size_m), (2,))
    ny, nx = modes.shape[-2:]
    y, x = np.meshgrid((np.arange(ny) - ny // 2) * dy,
                        (np.arange(nx) - nx // 2) * dx, indexing="ij")
    flat = modes.reshape(len(modes), -1)

    def moment(coordinate):
        operator = flat.conj() @ (flat * coordinate.ravel()).T
        return np.einsum("pij,ji->p", covariance, operator).real / total

    mx, my = moment(x), moment(y)
    vx, vy = np.maximum(0, moment(x*x) - mx*mx), np.maximum(0, moment(y*y) - my*my)
    return {
        "frame_indices": np.asarray(frame_indices)[order],
        "coherent_weights": coefficients, "mode_powers": mode_power,
        "mode_fractions": mode_power / total[:, None], "total_power": total,
        "purity": np.einsum("pij,pji->p", covariance, covariance).real / total**2,
        "centroids_m": np.column_stack((my, mx)),
        "d4sigma_m": 4 * np.sqrt(np.column_stack((vy, vx))),
    }


def _save(fig, path, filename_prefix=""):
    fig.savefig(path.with_name(filename_prefix + path.name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _plot_modes(raw, decomposition, output, pixel, filename_prefix=""):
    from ptycho_jax.visualisation.probe_plotting import probe_display_geometry
    n = len(decomposition["svd_modes"])
    dy, dx = pixel
    ny, nx = raw.shape[-2:]
    extent, xlabel, ylabel, size_label = probe_display_geometry((ny, nx), pixel)
    svd_power = np.mean(np.sum(abs(decomposition["svd_weights"]) ** 2, axis=-1), axis=0)
    fig, axes = plt.subplots(2, n, figsize=(max(5, 3*n), 6), squeeze=False, layout="constrained")
    for i, mode in enumerate(decomposition["svd_modes"]):
        weighted = mode * np.sqrt(svd_power[i])
        artist = axes[0, i].imshow(abs(weighted), cmap="viridis", extent=extent, origin="lower")
        fig.colorbar(artist, ax=axes[0, i], shrink=0.7)
        axes[0, i].set_title(f"SVD {i+1}: {100*svd_power[i]/svd_power.sum():.3f}%")
        axes[1, i].imshow(np.angle(mode), cmap="twilight", vmin=-np.pi, vmax=np.pi,
                           extent=extent, origin="lower")
        axes[1, i].set_title("Phase (rad)")
        for row in range(2):
            axes[row, i].set(xlabel=xlabel, ylabel=ylabel)
    axes[0, 0].set_ylabel("y (µm)")
    axes[1, 0].set_ylabel("y (µm)")
    fig.suptitle("SVD-orthogonalized spatial basis; transformed weights preserve every fitted field\n" + size_label)
    _save(fig, output / "probe_modes_svd.png", filename_prefix)

    fig, axes = plt.subplots(2, n, figsize=(max(5, 3*n), 6), squeeze=False, layout="constrained")
    for row, modes in enumerate((raw, decomposition["coherent_modes"])):
        for i, mode in enumerate(modes):
            scale = np.percentile(abs(mode), 99.5)
            axes[row, i].imshow(complex_beam_rgb(mode, amplitude_scale=scale),
                                  extent=extent, origin="lower")
            label = f"Fitted mode {i+1}" if row == 0 else (
                f"Coherent {i+1}: {100*decomposition['mode_fractions'][i]:.3f}%"
            )
            axes[row, i].set_title(label)
            axes[row, i].set(xlabel=xlabel, ylabel=ylabel)
    axes[0, 0].set_ylabel("y (µm)")
    axes[1, 0].set_ylabel("y (µm)")
    fig.suptitle("HSV: hue = phase; brightness = amplitude^0.5, scaled independently per mode\n"
                 "Top: fitted basis. Bottom: shot-ensemble coherent modes; constant phase is arbitrary.\n" + size_label)
    phase_key = plt.cm.ScalarMappable(norm=plt.Normalize(-np.pi, np.pi), cmap="hsv")
    fig.colorbar(phase_key, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.035,
                 pad=0.05, label="Phase (rad; black means zero amplitude)")
    _save(fig, output / "probe_modes_hsv.png", filename_prefix)

    fractions = decomposition["mode_fractions"]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.bar(np.arange(1, len(fractions)+1), fractions)
    for i, fraction in enumerate(fractions):
        ax.text(i+1, fraction, f"{100*fraction:.3f}%", ha="center", va="bottom")
    ax.set(xlabel="Ensemble coherent mode", ylabel="Fraction of mean beam power",
           title=f"Physical mode spectrum; purity = {np.sum(fractions**2):.5f}", ylim=(0, 1.12))
    _save(fig, output / "coherent_mode_spectrum.png", filename_prefix)


def _plot_propagation(propagation, nominal_defocus, output, filename_prefix=""):
    z = propagation["z_m"] * 1e3
    focus = propagation["focus_z_m"] * 1e3
    for coordinate, key, filename in (("x", "projection_x", "propagation_xz.png"),
                                        ("y", "projection_y", "propagation_yz.png")):
        axis = propagation[f"{coordinate}_m"] * 1e6
        marginal = propagation[key]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
        extent = (z[0], z[-1], axis[0], axis[-1])
        absolute = marginal.T / marginal.max()
        normalized = (marginal / marginal.max(axis=1, keepdims=True)).T
        for ax, image, title in ((axes[0], np.log10(np.maximum(absolute, 1e-6)), "Common power scale (log10)"),
                                 (axes[1], normalized, "Each plane normalized; reveals beam shape")):
            artist = ax.imshow(image, extent=extent, origin="lower", aspect="auto", cmap="magma")
            ax.axvline(0, color="white", ls=":", label="Sample")
            ax.axvline(-nominal_defocus*1e3, color="cyan", ls="--", label="Nominal focus")
            ax.axvline(focus, color="lime", ls="-", label="Estimated RMS focus")
            ax.set(xlabel="z from sample (mm; + downstream)", title=title)
            fig.colorbar(artist, ax=ax)
        axes[0].set_ylabel(f"{coordinate} (µm)")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"Near-field incoherent beam propagation: {coordinate}-z projection")
        _save(fig, output / filename, filename_prefix)

    planes = list(propagation["selected_planes"]) + [propagation["focus_plane"]]
    distances = list(propagation["selected_z_m"]) + [propagation["focus_z_m"]]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
    x, y = propagation["x_m"] * 1e6, propagation["y_m"] * 1e6
    extent = (x[0], x[-1], y[0], y[-1])
    for i, (ax, plane, distance) in enumerate(zip(axes.flat, planes, distances)):
        image = np.log10(np.maximum(plane / plane.max(), 1e-5))
        ax.imshow(image, extent=extent, origin="lower", cmap="magma", vmin=-5, vmax=0)
        ax.set(title=("RMS focus: " if i == len(planes)-1 else "") + f"z={distance*1e3:+.4f} mm",
               xlabel="x (µm)", ylabel="y (µm)")
    fig.suptitle("Propagated mean beam intensity; each plane normalized, log10 scale [-5,0]")
    _save(fig, output / "propagation_planes.png", filename_prefix)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, layout="constrained")
    axes[0].plot(z, propagation["d4sigma_x_m"]*1e6, label="x D4σ")
    axes[0].plot(z, propagation["d4sigma_y_m"]*1e6, label="y D4σ")
    for value, label, color in ((focus, "Joint RMS waist", "green"),
                                (propagation["focus_x_z_m"]*1e3, "x waist", "tab:blue"),
                                (propagation["focus_y_z_m"]*1e3, "y waist", "tab:orange"),
                                (-nominal_defocus*1e3, "Nominal focus", "black")):
        axes[0].axvline(value, ls="--", color=color, label=f"{label}: {value:+.4f} mm")
    axes[0].set(ylabel="D4σ beam diameter (µm)", title="Estimated beam waists from propagated second moments")
    axes[0].legend(fontsize=8)
    axes[1].plot(z, propagation["effective_area_m2"]*1e12, color="tab:purple", label="Effective beam area")
    axes[1].axvline(propagation["effective_area_focus_z_m"]*1e3, color="tab:purple", ls="--",
                    label=f"Minimum area: {propagation['effective_area_focus_z_m']*1e3:+.4f} mm")
    axes[1].set(xlabel="z from sample (mm; + downstream)", ylabel="(∫I)²/∫I² (µm²)")
    axes[1].legend()
    _save(fig, output / "focus_estimate.png", filename_prefix)


def _plot_scan(scan, output, filename_prefix=""):
    initial, corrected, correction = scan["initial_m"]*1e6, scan["corrected_m"]*1e6, scan["corrections_m"]*1e6
    shots = scan["frame_indices"] + 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    axes[0].scatter(initial[:, 1], initial[:, 0], s=35, facecolors="none", edgecolors="gray", label="Initial")
    artist = axes[0].scatter(corrected[:, 1], corrected[:, 0], c=shots, s=10, cmap="viridis", label="Corrected")
    fig.colorbar(artist, ax=axes[0], label="Acquisition frame number")
    axes[0].legend()
    axes[0].set_title("Initial and corrected spots: actual coordinates")
    axes[1].scatter(initial[:, 1], initial[:, 0], c=shots, s=10, cmap="viridis")
    axes[1].quiver(initial[:, 1], initial[:, 0], correction[:, 1]*20, correction[:, 0]*20,
                    angles="xy", scale_units="xy", scale=1, width=0.003)
    axes[1].set_title("Correction vectors magnified 20×")
    for ax in axes:
        ax.set(xlabel="x (µm)", ylabel="y (µm)")
        ax.set_aspect("equal")
    _save(fig, output / "scan_positions.png", filename_prefix)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, component, label in ((axes[0], 1, "x"), (axes[1], 0, "y")):
        values = scan["corrections_m"][:, component]*1e9
        ax.plot(shots, values, ".-", lw=0.6, ms=3, label="Fitted correction")
        ax.plot(shots, scan["drift_trend_m"][:, component]*1e9, "--", color="red",
                 label=f"Linear trend {scan['drift_slope_m_per_shot'][component]*1e9:+.4f} nm/shot; "
                       f"R²={scan['drift_r_squared'][component]:.3f}")
        ax.set_ylabel(f"Δ{label} (nm)")
        ax.legend(fontsize=8)
    axes[2].plot(shots, np.linalg.norm(scan["corrections_m"], axis=1)*1e9, color="black")
    axes[2].set(xlabel="Acquisition frame number (1-based, original HDF5 order)", ylabel="Correction magnitude (nm)")
    fig.suptitle("Position corrections versus shot order: descriptive drift check, not elapsed time")
    _save(fig, output / "position_drift.png", filename_prefix)


def _plot_modal_evolution(raw_weights, evolution, output, filename_prefix=""):
    shots = evolution["frame_indices"] + 1
    n, m = raw_weights.shape[1:]
    flat = raw_weights.reshape(len(shots), -1)
    labels = [f"{i+1}→{j+1}" for i in range(n) for j in range(m)]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), layout="constrained")
    # Images use row index when shot IDs have gaps; explicitly label sampled IDs.
    extent = (-0.5, n*m-0.5, -0.5, len(shots)-0.5)
    for ax, image, cmap, title in ((axes[0], abs(flat), "viridis", "Raw |W[n,m]|"),
                                    (axes[1], np.where(abs(flat)>abs(flat).max()*0.01, np.angle(flat), np.nan),
                                     "twilight", "Raw arg(W): shown above 1% amplitude; basis dependent")):
        artist = ax.imshow(image, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.set_xticks(range(n*m), labels, rotation=90, fontsize=7)
        ticks = np.linspace(0, len(shots)-1, min(6, len(shots))).astype(int)
        ax.set_yticks(ticks, shots[ticks])
        ax.set(ylabel="Acquisition frame", title=title)
        fig.colorbar(artist, ax=ax, shrink=0.8)
    for i in range(min(n, m)):
        axes[2].plot(shots, abs(raw_weights[:, i, i]), label=f"|W[{i+1},{i+1}]|")
    axes[2].set(xlabel="Acquisition frame number", ylabel="Diagonal amplitude weights")
    axes[2].legend(ncol=min(n, 5))
    _save(fig, output / "modal_weights_raw.png", filename_prefix)

    fractions = evolution["mode_fractions"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    labels = [f"Coherent mode {i+1}" for i in range(fractions.shape[1])]
    axes[0].stackplot(shots, fractions.T, labels=labels)
    axes[0].set(ylabel="Power fraction", ylim=(0, 1), title="Fixed ensemble-orthogonal basis: mode populations by shot")
    axes[0].legend(ncol=3, fontsize=8, loc="lower right")
    for i, label in enumerate(labels):
        axes[1].semilogy(shots, np.maximum(fractions[:, i], 1e-12), label=label)
    axes[1].set_ylabel("Power fraction (log scale)")
    axes[2].plot(shots, evolution["total_power"] / np.mean(evolution["total_power"]), label="Total beam power / mean")
    axes[2].plot(shots, evolution["purity"], label="Per-shot coherence purity")
    axes[2].set(xlabel="Acquisition frame number (1-based)", ylabel="Normalized power / purity")
    axes[2].legend()
    _save(fig, output / "modal_weights_evolution.png", filename_prefix)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    centroids = evolution["centroids_m"]
    for component, label in ((1, "x"), (0, "y")):
        axes[0].plot(shots, (centroids[:, component] - centroids[:, component].mean())*1e9,
                       label=f"{label} centroid relative to mean")
        axes[1].plot(shots, evolution["d4sigma_m"][:, component]*1e6, label=f"{label} D4σ")
    axes[0].set(ylabel="Centroid displacement (nm)", title="Fitted beam evolution at the sample plane")
    axes[1].set(xlabel="Acquisition frame number (1-based)", ylabel="D4σ diameter (µm)")
    for ax in axes:
        ax.legend()
    _save(fig, output / "beam_evolution.png", filename_prefix)


def write_diagnostics(mode_folder, *, half_range_m=None, z_samples=201, padding_factor=3,
                      filename_prefix="") -> dict:
    """Analyze saved parameters; optional prefix applies to all inputs/outputs.

    Orthogonalization transforms analysis copies and compensates shot weights;
    it preserves fitted fields and does not overwrite optimized parameters.
    """
    folder = Path(mode_folder)
    output = folder / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    with np.load(folder / f"{filename_prefix}reconstruction.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    metrics = json.loads((folder / f"{filename_prefix}metrics.json").read_text())
    geometry = metrics["geometry"]
    nominal = geometry["assumed_defocus_at_sample_m"]
    if half_range_m is None:
        half_range_m = 2 * abs(nominal)
    pixel = data["sample_pixel_size_m"]
    decomposition = orthogonalize_probe_ensemble(data["probe_modes"], data["modal_weights"])
    print("Diagnostics: orthogonalization complete; propagating mean beam", flush=True)
    propagation = analyze_propagation(
        decomposition["coherent_modes"], decomposition["mode_powers"], pixel,
        geometry["wavelength_m"], half_range_m=half_range_m,
        z_samples=z_samples, padding_factor=padding_factor,
    )
    scan = analyze_scan_positions(data["scan_coords_pix"], data["position_corrections_pix"],
                                  data["frame_indices"], pixel)
    evolution = analyze_modal_evolution(decomposition, data["frame_indices"], pixel)
    _plot_modes(data["probe_modes"], decomposition, output, pixel, filename_prefix)
    _plot_propagation(propagation, nominal, output, filename_prefix)
    _plot_scan(scan, output, filename_prefix)
    _plot_modal_evolution(data["modal_weights"][scan["order"]], evolution, output, filename_prefix)
    relative_rms = np.sqrt(np.mean(scan["corrections_m"]**2, axis=0))
    summary = {
        "estimated_focus_z_m": propagation["focus_z_m"],
        "estimated_focus_to_sample_m": -propagation["focus_z_m"],
        "nominal_focus_z_m": -nominal,
        "focus_x_z_m": propagation["focus_x_z_m"], "focus_y_z_m": propagation["focus_y_z_m"],
        "effective_area_focus_z_m": propagation["effective_area_focus_z_m"],
        "focus_at_boundary": propagation["focus_at_boundary"],
        "focus_d4sigma_x_m": propagation["focus_d4sigma_x_m"],
        "focus_d4sigma_y_m": propagation["focus_d4sigma_y_m"],
        "propagation_half_range_m": half_range_m, "z_samples": z_samples,
        "z_step_m": float(propagation["z_m"][1]-propagation["z_m"][0]),
        "padding_factor": padding_factor,
        "maximum_border_energy_fraction": float(np.max(propagation["border_energy_fraction"])),
        "focus_border_energy_fraction": float(np.interp(propagation["focus_z_m"], propagation["z_m"],
                                                          propagation["border_energy_fraction"])),
        "maximum_propagation_energy_relative_error": propagation["maximum_energy_relative_error"],
        "ensemble_mode_power_fractions": decomposition["mode_fractions"],
        "ensemble_coherence_purity": float(np.sum(decomposition["mode_fractions"]**2)),
        "position_correction_rms_m_yx": relative_rms,
        "position_drift_slope_m_per_shot_yx": scan["drift_slope_m_per_shot"],
        "position_drift_r_squared_yx": scan["drift_r_squared"],
        "position_drift_residual_rms_m_yx": scan["drift_residual_rms_m"],
        "beam_centroid_std_m_yx": np.std(evolution["centroids_m"], axis=0),
        "beam_d4sigma_mean_m_yx": np.mean(evolution["d4sigma_m"], axis=0),
        "acquisition_frame_indices": scan["frame_indices"],
        "references": {"Fresnel propagation": PROPAGATION_SOURCE, "D4sigma definition": BEAM_WIDTH_SOURCE},
    }
    (output / f"{filename_prefix}summary.json").write_text(json.dumps(summary, default=_json_default, indent=2, allow_nan=False) + "\n")
    arrays = {}
    for prefix, values in (("basis", decomposition), ("propagation", propagation), ("scan", scan), ("evolution", evolution)):
        for key, value in values.items():
            arrays[f"{prefix}_{key}"] = value
    np.savez_compressed(output / f"{filename_prefix}diagnostics.npz", **arrays)
    _write_report(output, summary, filename_prefix)
    print(f"Diagnostics: RMS focus z={summary['estimated_focus_z_m']*1e3:+.5f} mm; saved {output}", flush=True)
    return summary


def _write_report(output, summary, filename_prefix=""):
    s = summary
    images = [
        ("SVD-orthogonalized modes", "probe_modes_svd.png"),
        ("HSV complex beam modes", "probe_modes_hsv.png"),
        ("Ensemble coherent mode spectrum", "coherent_mode_spectrum.png"),
        ("Horizontal propagation", "propagation_xz.png"),
        ("Vertical propagation", "propagation_yz.png"),
        ("Propagated intensity planes", "propagation_planes.png"),
        ("Estimated focus and astigmatism", "focus_estimate.png"),
        ("Initial and corrected scan spots", "scan_positions.png"),
        ("Position correction evolution", "position_drift.png"),
        ("Raw modal-weight evolution", "modal_weights_raw.png"),
        ("Orthogonal-basis modal populations", "modal_weights_evolution.png"),
        ("Beam centroid and width evolution", "beam_evolution.png"),
    ]
    report = [
        "# NanoMAX beam diagnostics", "",
        f"Estimated joint RMS focus: **z={s['estimated_focus_z_m']*1e3:+.6f} mm from the sample**; "
        f"positive z is downstream. Estimated focus-to-sample distance: {s['estimated_focus_to_sample_m']*1e3:+.6f} mm.",
        f"Nominal focus: z={s['nominal_focus_z_m']*1e3:+.6f} mm. "
        f"Horizontal and vertical waists: {s['focus_x_z_m']*1e3:+.6f} and {s['focus_y_z_m']*1e3:+.6f} mm.",
        f"Effective-area minimum (core-sensitive check): z={s['effective_area_focus_z_m']*1e3:+.6f} mm.",
        f"D4σ diameters at joint RMS focus: x={s['focus_d4sigma_x_m']*1e9:.2f} nm, "
        f"y={s['focus_d4sigma_y_m']*1e9:.2f} nm.",
        f"Sweep: ±{s['propagation_half_range_m']*1e3:g} mm; {s['z_samples']} planes; "
        f"step {s['z_step_m']*1e6:g} µm; zero-padding factor {s['padding_factor']}.",
        f"Focus at search boundary: {s['focus_at_boundary']}. "
        f"Maximum outer-5%-border power fraction: {s['maximum_border_energy_fraction']:.4%}; "
        f"near focus: {s['focus_border_energy_fraction']:.4%}. "
        f"Maximum propagation energy error: {s['maximum_propagation_energy_relative_error']:.3g}.", "",
        "## Mode spectrum", "", "| Mode | Ensemble mean power |", "| --- | --- |",
    ]
    for i, fraction in enumerate(s["ensemble_mode_power_fractions"]):
        report.append(f"| {i+1} | {100*fraction:.6f}% |")
    report += ["", f"Ensemble coherence purity: {s['ensemble_coherence_purity']:.6f}.", "",
        "## Acquisition-order position trends", "",
        "All evolution plots use original HDF5 frame indices, not coordinate-sorted reconstruction order.",
        f"Correction RMS x/y: {s['position_correction_rms_m_yx'][1]*1e9:.3f} / "
        f"{s['position_correction_rms_m_yx'][0]*1e9:.3f} nm.",
        f"Linear correction trends x/y: {s['position_drift_slope_m_per_shot_yx'][1]*1e9:+.5f} / "
        f"{s['position_drift_slope_m_per_shot_yx'][0]*1e9:+.5f} nm per shot; "
        f"R²: {s['position_drift_r_squared_yx'][1]:.4f} / {s['position_drift_r_squared_yx'][0]:.4f}.", "",
        "These are descriptive trends. Shot number is not elapsed time; scan-dependent model errors "
        "can resemble drift. No claim of a measured temporal drift rate is made.", "",
        "## Analysis conventions and limits", "",
        "The raw complex basis is SVD-orthogonalized with its full weight matrices transformed to preserve "
        "every coherent shot field. The covariance of these coefficients is diagonalized to obtain "
        "shot-averaged coherent modes and gauge-independent mean power fractions. No dense spatial "
        "mutual-intensity matrix is allocated. SVD-basis powers and ensemble coherent-mode powers differ "
        "when the SVD-basis coefficient covariance has off-diagonal terms.", "",
        "HSV hue encodes phase and brightness encodes amplitude to power 0.5. Each mode's brightness "
        "is normalized independently for visibility; use the spectrum for power comparisons. The constant "
        "phase of an individual orthogonal mode is arbitrary. Raw modal weights depend on the fitted basis "
        "and incoherent channel gauge; orthogonal-basis population and beam-moment plots are more physical.", "",
        "Propagation uses the paraxial transfer function exp[-i π λ z (fx²+fy²)] at the reconstructed "
        "sample pixel sizes. Orthogonal modes are propagated coherently one at a time and their intensities "
        "are summed with ensemble powers. The common carrier phase is omitted. Zero padding reduces "
        "periodic wraparound; edge-power fractions and energy checks are recorded. Energy conservation "
        "alone does not rule out wraparound. Focus minimizes the centered second-moment trace "
        "σx²+σy²; x/y waists and the inverse-participation-area minimum are additional diagnostics. "
        "Three neighboring z samples refine each minimum quadratically; this is not a statistical uncertainty.", "",
        "The estimated focus is conditional on the reconstructed wavefront, detector crop, pixel calibration "
        "and finite field of view. It is not an independently calibrated true focus or the objective's focal "
        "length. Broad halos can affect second moments; compare the core-sensitive focus metric. "
        "Object/probe ambiguities and weak modes also limit physical interpretation.", "",
        f"Propagation reference: [TU Delft Advanced Optical Imaging]({PROPAGATION_SOURCE}). "
        f"Width reference: [NIST SP 250-87]({BEAM_WIDTH_SOURCE}).", "",
    ]
    for title, filename in images:
        report += [f"## {title}", "", f"![{title}]({filename_prefix}{filename})", ""]
    report += [f"Numeric data: [diagnostics.npz]({filename_prefix}diagnostics.npz); "
               f"[summary.json]({filename_prefix}summary.json).", ""]
    (output / f"{filename_prefix}results.md").write_text("\n".join(report))
