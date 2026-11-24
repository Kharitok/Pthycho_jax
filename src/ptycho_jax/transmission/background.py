"""One nonnegative detector background shape with a bounded level per frame.

The shared probability image sums to one on valid detector pixels. A frame's
background photon count is its measured total times a bounded fitted fraction.
Thus the map/scale ambiguity is removed and a photon budget is enforced by
parameterization, independently of optional post-update projections.
"""

import jax
import jax.numpy as jnp
import numpy as np


def initialize_background(
    mask, frame_totals, grid_shape=(16, 16), initial_fraction=1e-4, max_fraction=0.05
):
    """Return real optimizer leaves and fixed data; coarse logits are centered."""
    mask = np.asarray(mask, dtype=bool)
    totals = np.asarray(frame_totals)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Background mask must be a nonempty 2D validity image.")
    if (
        totals.ndim != 1
        or not len(totals)
        or not np.isfinite(totals).all()
        or np.any(totals <= 0)
    ):
        raise ValueError("Background frame totals must be finite, positive and 1D.")
    if len(grid_shape) != 2 or any(type(n) is not int or n < 2 for n in grid_shape):
        raise ValueError("Background grid must contain two integers >=2.")
    if not 0 < initial_fraction < max_fraction < 1:
        raise ValueError(
            "Require 0 < background_initial_fraction < background_max_fraction < 1."
        )
    ratio = initial_fraction / max_fraction
    logit = np.log(ratio) - np.log1p(-ratio)
    return {
        "background_logits": jnp.zeros(tuple(grid_shape), dtype=jnp.float32),
        "background_scale_logits": jnp.full(totals.shape, logit, dtype=jnp.float32),
    }, {
        "background_mask": jnp.asarray(mask),
        "background_frame_totals": jnp.asarray(totals, dtype=jnp.float32),
        "background_max_fraction": jnp.asarray(max_fraction, dtype=jnp.float32),
    }


def background_probability(params, fixed):
    """Shared detector image in unshifted FFT order, normalized on valid pixels.

    Bilinear interpolation of coarse, detector-centered logits restricts shape
    complexity. Softmax is nonnegative and fixes the map normalization. Masked
    pixels have exactly zero probability and never influence normalization.
    """
    mask = fixed["background_mask"]
    centered = jax.image.resize(
        params["background_logits"], mask.shape, method="linear"
    )
    logits = jnp.fft.ifftshift(centered)
    logits = jnp.where(mask, logits, -jnp.inf)
    return jax.nn.softmax(logits.reshape(-1)).reshape(mask.shape)


def background_fractions(params, fixed):
    """Fraction of measured valid-pixel photons assigned to each background."""
    return fixed["background_max_fraction"] * jax.nn.sigmoid(
        params["background_scale_logits"]
    )


def background_frame_counts(params, fixed):
    return fixed["background_frame_totals"] * background_fractions(params, fixed)


def shared_scaled_background(params, fixed, pos_index):
    """Additive counts in binned detector pixels for a single frame."""
    return background_frame_counts(params, fixed)[pos_index] * background_probability(
        params, fixed
    )


def background_components(params, fixed):
    """Export physical arrays with mean(frame scales)=1 for interpretation.

    map is the mean additive intensity in counts/pixel. frame_counts[j] times
    probability equals scales[j] times map, including for a vanishing level.
    """
    probability = np.asarray(background_probability(params, fixed))
    counts = np.asarray(background_frame_counts(params, fixed))
    mean = counts.mean(dtype=float)
    return {
        "map": probability * mean,
        "scales": counts / mean if mean > 0 else np.ones_like(counts),
        "frame_counts": counts,
        "probability": probability,
        "fractions": np.asarray(background_fractions(params, fixed)),
    }


def audit_background(params, fixed, measured_mean, prediction_mean, prediction_totals):
    """Measure photon fractions and bright-region leakage, without hiding pixels."""
    components = background_components(params, fixed)
    mask = np.asarray(fixed["background_mask"])
    counts = components["frame_counts"].astype(float)
    total_measured = np.asarray(fixed["background_frame_totals"], dtype=float)
    prediction_totals = np.asarray(prediction_totals, dtype=float)
    bright = mask & (measured_mean >= np.percentile(measured_mean[mask], 90))
    mean_bg = components["map"]
    mean_signal = prediction_mean - mean_bg
    bright_fraction = float(
        mean_bg[bright].sum(dtype=float)
        / max(prediction_mean[bright].sum(dtype=float), 1e-20)
    )
    fraction_prediction = counts / np.maximum(prediction_totals, 1e-20)
    global_fraction = float(counts.sum() / max(prediction_totals.sum(), 1e-20))
    audit = {
        "enabled": True,
        "mean_measured_fraction": float(counts.sum() / total_measured.sum()),
        "max_frame_measured_fraction": float(np.max(counts / total_measured)),
        "mean_predicted_fraction": global_fraction,
        "max_frame_predicted_fraction": float(fraction_prediction.max()),
        "bright_pixel_predicted_fraction": bright_fraction,
        "bright_region_definition": "top 10% of valid pixels by mean measured intensity",
        "fraction_valid_pixels_background_above_signal": float(
            np.mean(mean_bg[mask] > mean_signal[mask])
        ),
        "fraction_frames_near_budget": float(
            np.mean(
                components["fractions"]
                >= 0.95 * float(fixed["background_max_fraction"])
            )
        ),
        "max_fraction_budget": float(fixed["background_max_fraction"]),
        "mean_counts_per_valid_pixel": float(counts.mean() / mask.sum()),
        "scale_min": float(components["scales"].min()),
        "scale_max": float(components["scales"].max()),
        "dominates_signal": bool(
            global_fraction >= 0.5
            or fraction_prediction.max() >= 0.5
            or bright_fraction >= 0.5
        ),
    }
    return components, audit
