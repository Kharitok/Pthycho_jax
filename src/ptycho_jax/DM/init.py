"""Seeded unit-mode initialization with exact masked-count calibration."""

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter
from .types import DMState, Data
from .stages import exit_waves, forward_from_stages


def make_data(intensities, mask, dtype=None):
    """Finite nonnegative intensity (J,Y,X), broadcast mask -> Data with amplitudes."""
    values = np.asarray(intensities)
    if values.ndim != 3 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected finite nonnegative intensities (shots,Y,X).")
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), values.shape)
    if np.any(mask.sum((-2, -1)) == 0):
        raise ValueError("Every shot needs valid detector pixels.")
    arr = jnp.asarray(values, dtype=dtype)
    mm = jnp.asarray(mask)
    return Data(jnp.sqrt(arr), mm, jnp.sum(jnp.where(mm, arr, 0), axis=(-2, -1)))


def initialize_modes(probe0, n_modes, seed=0):
    """Supplied field (Y,X) -> seeded orthonormal complex modes (K,Y,X)."""
    p0 = np.asarray(probe0)
    if p0.ndim != 2 or n_modes < 1 or n_modes > p0.size or np.linalg.norm(p0) == 0:
        raise ValueError("Expected nonzero 2D probe and 1<=n_modes<=pixels.")
    dtype = np.complex64 if p0.dtype == np.complex64 else np.complex128
    rng = np.random.default_rng(seed)
    candidates = [p0]
    for k in range(1, n_modes):
        noise = gaussian_filter(rng.normal(size=p0.shape), 2) + 1j * gaussian_filter(
            rng.normal(size=p0.shape), 2
        )
        candidates.append(p0 * (1 + 0.02 * noise))
    q, _ = np.linalg.qr(np.stack(candidates).reshape(n_modes, -1).T)
    modes = q.T.reshape((n_modes, *p0.shape))
    phase = np.vdot(modes[0], p0)
    modes[0] *= phase / abs(phase)
    return modes.astype(dtype)


def initialize_state(probe0, n_modes, data, geom, *, seed=0, profile=None):
    """Unit probe (K,Y,X), ones canvas, calibrated a (J,K), waves -> DMState."""
    probe = jnp.asarray(initialize_modes(probe0, n_modes, seed))
    obj = jnp.ones(geom.canvas_shape, dtype=probe.dtype)
    power = np.asarray(
        [1 / (10**k) for k in range(n_modes)] if profile is None else profile,
        dtype=float,
    )
    if power.shape != (n_modes,) or np.any(power < 0) or power.sum() <= 0:
        raise ValueError(
            "Mode power profile must have K nonnegative entries and positive total."
        )
    power = jnp.asarray(power / power.sum(), dtype=probe.real.dtype)
    a = jnp.sqrt(data.counts[:, None] * power[None])
    intensity = forward_from_stages(probe, obj, a, geom)
    counts = jnp.sum(jnp.where(data.mask, intensity, 0), axis=(-2, -1))
    factor = jnp.sqrt(data.counts / jnp.maximum(counts, jnp.finfo(counts.dtype).tiny))
    a = a * factor[:, None]
    return DMState(
        exit_waves(probe, obj, a, geom), probe, obj, a, jnp.array(0, jnp.int32)
    )
