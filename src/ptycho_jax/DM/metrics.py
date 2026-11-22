"""Model-error reduction exactly matching the reference Gaussian amplitude loss."""

import jax.numpy as jnp
from .stages import detector_intensity
from .conventions import fft2c


def amplitude_metrics(psi, data):
    """Waves (J,K,Y,X) -> scalar masked amplitude MSE and count-normalized R."""
    intensity = detector_intensity(fft2c(psi))
    residual = (
        jnp.sqrt(intensity + jnp.asarray(1e-20, intensity.dtype)) - data.sqrt_I
    ) ** 2
    sums = jnp.sum(jnp.where(data.mask, residual, 0), axis=(-2, -1))
    counts = jnp.sum(data.mask, axis=(-2, -1))
    mse = jnp.mean(sums / counts)
    r = jnp.sum(sums) / jnp.maximum(
        jnp.sum(data.counts), jnp.finfo(intensity.dtype).tiny
    )
    return {"model_mse": mse, "model_r": r}


def aligned_object_error(recovered, truth, covered):
    """Gauge-aligned relative object error over a boolean (H,W) support.

    Remove constant amplitude/phase and a fitted affine phase difference. This
    metric does not resolve translation; the ladder separately records it.
    """
    import numpy as np

    rec = np.asarray(recovered, dtype=np.complex128).copy()
    true = np.asarray(truth, dtype=np.complex128)
    mask = np.asarray(covered)
    from scipy.optimize import minimize

    y, x = np.indices(rec.shape)
    delta = rec * np.conj(true)
    my = mask[1:] & mask[:-1]
    mx = mask[:, 1:] & mask[:, :-1]
    ay = np.angle(np.sum((delta[1:] * np.conj(delta[:-1]))[my]))
    ax = np.angle(np.sum((delta[:, 1:] * np.conj(delta[:, :-1]))[mx]))
    rv, tv = rec[mask], true[mask]
    yy, xx = y[mask], x[mask]
    denominator = np.vdot(tv, tv).real

    def error(slopes):
        shifted = rv * np.exp(-1j * (slopes[0] * yy + slopes[1] * xx))
        scalar = np.vdot(shifted, tv) / np.vdot(shifted, shifted)
        return float(np.sum(abs(scalar * shifted - tv) ** 2) / denominator)

    fit = minimize(error, [ay, ax], method="BFGS", options={"gtol": 1e-10})
    slopes = fit.x if error(fit.x) < error([ay, ax]) else [ay, ax]
    rec *= np.exp(-1j * (slopes[0] * y + slopes[1] * x))
    scalar = np.vdot(rec[mask], true[mask]) / np.vdot(rec[mask], rec[mask])
    return float(
        np.linalg.norm((scalar * rec - true)[mask]) / np.linalg.norm(true[mask])
    )


def coherence_overlap(p1, w1, p2, w2):
    """Coherence-operator overlap from (K,Y,X) modes and mean intensity weights."""
    import numpy as np

    f1 = np.asarray(p1).reshape(len(p1), -1)
    f2 = np.asarray(p2).reshape(len(p2), -1)

    def inner(x, a, y, b):
        return np.sum(a[:, None] * b[None] * abs(x.conj() @ y.T) ** 2)

    numerator = inner(f1, np.asarray(w1), f2, np.asarray(w2))
    return float(numerator / np.sqrt(inner(f1, w1, f1, w1) * inner(f2, w2, f2, w2)))
