"""Show additive detector counts and frame scaling separately from diffraction."""

import matplotlib.pyplot as plt
import numpy as np


def save_background_figures(
    components, data, final, folder, filename_prefix, max_fraction
):
    mask = np.asarray(data["detector_mask"])
    mean_bg = components["mean_observed_map"]
    signal = np.maximum(final["mean_prediction"] - mean_bg, 0)
    fraction = mean_bg / np.maximum(final["mean_prediction"], 1e-20)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, image, title, limits in (
        (axes[0], np.log10(1 + signal), "Mean diffraction signal: log10(1+counts)", {}),
        (
            axes[1],
            np.log10(1 + mean_bg),
            "Mean additive background: log10(1+counts)",
            {},
        ),
        (
            axes[2],
            fraction,
            "Background / mean predicted intensity",
            {"vmin": 0, "vmax": 1},
        ),
    ):
        artist = ax.imshow(
            np.fft.fftshift(np.where(mask, image, np.nan)),
            origin="lower",
            cmap="magma",
            **limits,
        )
        ax.set(
            title=title,
            xlabel="Detector column (pixels)",
            ylabel="Detector row (pixels)",
        )
        fig.colorbar(artist, ax=ax, shrink=0.8)
    fig.savefig(
        folder / f"{filename_prefix}background.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    order = np.argsort(data["frame_indices"], kind="stable")
    shots = np.asarray(data["frame_indices"])[order] + 1
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    axes[0].plot(shots, components["scales"][order])
    axes[0].set(
        ylabel="Background scale (mean = 1)",
        title="Shared background with per-frame intensity scaling",
    )
    axes[1].plot(
        shots,
        100 * components["fractions"][order],
        label="Fraction of measured photons",
    )
    axes[1].plot(
        shots,
        100 * (components["observed_frame_counts"] / final["predicted_totals"])[order],
        label="Fraction of predicted photons",
    )
    axes[1].axhline(
        100 * max_fraction, color="red", ls=":", label="Measured-photon budget cap"
    )
    axes[1].set(xlabel="Acquisition frame number", ylabel="Background photons (%)")
    axes[1].legend()
    fig.savefig(
        folder / f"{filename_prefix}background_scaling.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
