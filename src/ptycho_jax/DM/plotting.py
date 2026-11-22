"""Standalone snapshots and convergence figures for DM-to-AD experiments."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_snapshot(
    folder, params, fixed, pixel, origin, *, title, frame_indices=None, margin=10
):
    """Save complex parameters and object/probe/weight PNGs for one solver stage."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    p = np.asarray(params["probe_modes"])
    obj = np.asarray(params["sample"])
    weights = np.asarray(params["modal_weights"])
    pos = np.asarray(fixed["sample_positions"])
    np.savez_compressed(
        folder / "parameters.npz",
        **{k: np.asarray(v) for k, v in params.items()},
        fixed_sample_positions=pos,
        fixed_scan_mistakes=np.asarray(fixed["scan_mistakes"]),
        sample_pixel_size_m=pixel,
        object_pixel_origin=origin,
        frame_indices=np.arange(len(pos)) if frame_indices is None else frame_indices,
    )
    covariance = np.einsum("pnm,pkm->nk", weights, weights.conj()) / len(weights)
    intensity = np.einsum("nk,nyx,kyx->yx", covariance, p, p.conj()).real
    coverage = np.zeros(obj.shape)
    for y, x in pos:
        y += margin
        x += margin
        coverage[y : y + p.shape[1], x : x + p.shape[2]] += np.maximum(intensity, 0)
    mask = coverage > coverage.max() * 0.02
    yy, xx = np.nonzero(mask)
    roi = (slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))
    dy, dx = np.asarray(pixel) * 1e6
    oy, ox = origin
    extent = (
        (roi[1].start - ox) * dx,
        (roi[1].stop - ox) * dx,
        (roi[0].start - oy) * dy,
        (roi[0].stop - oy) * dy,
    )
    phase = np.angle(obj)
    low, high = np.percentile(phase[mask], (1, 99))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, image, cmap, label in (
        (axes[0], abs(obj), "gray", "Object amplitude"),
        (axes[1], phase, "twilight", "Object phase"),
    ):
        kwargs = (
            {"vmin": low, "vmax": high} if cmap == "twilight" and high > low else {}
        )
        artist = ax.imshow(
            np.where(mask, image, np.nan)[roi],
            origin="lower",
            extent=extent,
            cmap=cmap,
            **kwargs,
        )
        ax.set(title=label, xlabel="x (µm)", ylabel="y (µm)")
        fig.colorbar(artist, ax=ax)
    fig.suptitle(title)
    fig.savefig(folder / "sample.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(
        2, len(p), figsize=(max(6, 3 * len(p)), 6), squeeze=False, layout="constrained"
    )
    for k, mode in enumerate(p):
        artist = axes[0, k].imshow(abs(mode), cmap="viridis")
        fig.colorbar(artist, ax=axes[0, k], shrink=0.7)
        axes[0, k].set_title(f"Probe {k + 1} amplitude")
        axes[1, k].imshow(np.angle(mode), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1, k].set_title("Phase (rad)")
    fig.suptitle(title)
    fig.savefig(folder / "probe.png", dpi=150)
    plt.close(fig)
    order = np.arange(len(pos)) if frame_indices is None else np.argsort(frame_indices)
    shot = order if frame_indices is None else np.asarray(frame_indices)[order]
    power = np.sum(abs(weights) ** 2, axis=2) * np.sum(abs(p) ** 2, axis=(1, 2))[None]
    fractions = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-30)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for k in range(len(p)):
        ax.plot(shot + 1, fractions[order, k], label=f"Fitted basis mode {k + 1}")
    ax.set(
        xlabel="Original acquisition frame",
        ylabel="Diagonal-basis power proxy",
        title=title,
    )
    ax.legend()
    fig.savefig(folder / "weights.png", dpi=150)
    plt.close(fig)


def save_convergence(folder, dm_history, ad_history, baseline_history, metrics):
    """Plot per-iteration DM diagnostics and the exact DM/AD handoff objective."""
    folder = Path(folder)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    for key, label in (
        ("fourier_error", "Iterate Fourier R"),
        ("model_r", "Model R"),
        ("update_norm", "DM update norm"),
    ):
        axes[0].semilogy(
            np.arange(1, len(dm_history[key]) + 1),
            np.maximum(dm_history[key], 1e-20),
            label=label,
        )
    axes[0].set(
        xlabel="DM iteration",
        ylabel="Normalized error",
        title="Difference-map monitoring",
    )
    axes[0].legend()
    n = len(dm_history["model_mse"])
    axes[1].semilogy(
        np.arange(n + 1),
        np.r_[metrics["initial_mse"], dm_history["model_mse"]],
        label="DM overlap model",
    )
    axes[1].semilogy(
        n + np.arange(len(ad_history) + 1),
        np.r_[ad_history, metrics["dm_ad"]["final_loss"]],
        label="AD from DM",
    )
    axes[1].axvline(n, ls="--", color="gray", label="DM → AD")
    axes[1].set(
        xlabel="Stage iteration/epoch",
        ylabel="Masked amplitude MSE",
        title="Continuous handoff",
    )
    axes[1].legend()
    axes[2].semilogy(
        np.arange(len(ad_history) + 1),
        np.r_[ad_history, metrics["dm_ad"]["final_loss"]],
        label="AD after DM",
    )
    axes[2].semilogy(
        np.arange(len(baseline_history) + 1),
        np.r_[baseline_history, metrics["ad_only"]["final_loss"]],
        label="AD-only control",
    )
    axes[2].set(
        xlabel="AD epochs (equal counts)",
        ylabel="Masked amplitude MSE",
        title="Warm versus naive AD starts",
    )
    axes[2].legend()
    fig.savefig(folder / "convergence.png", dpi=150)
    plt.close(fig)
