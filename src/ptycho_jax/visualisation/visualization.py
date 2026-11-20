import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


def complex_to_hsv(field, sat_power=1.0, value=1.0, modulus_percentile=99.5):
    """
    Map a complex 2D array to RGB using:
        hue        = phase   (angle, wrapped to [0, 1))
        saturation = modulus (normalized to [0, 1])
        value      = constant (or pass an array to also modulate brightness)

    field: complex array [Y, X]
    sat_power: gamma applied to normalized modulus before using as saturation
               (values <1 boost low-amplitude regions' visibility)
    modulus_percentile: normalize modulus by this percentile instead of the
               max, so a few hot pixels don't wash out the whole scale
    """
    amp = np.abs(field)
    phase = np.angle(field)  # [-pi, pi]

    hue = (phase + np.pi) / (2 * np.pi)  # -> [0, 1)

    denom = np.percentile(amp, modulus_percentile)
    denom = denom if denom > 0 else 1.0
    sat = np.clip(amp / denom, 0, 1) ** sat_power

    val = np.full_like(amp, value, dtype=float)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb


def orthogonalize_modes(modes: jnp.ndarray) -> jnp.ndarray:
    """
    Orthogonalizes a set of spatial modes using the Gram-Schmidt process.

    Args:
        modes: Input modes [N, Y, X]

    Returns:
        Orthogonalized modes [N, Y, X]
    """
    N, Y, X = modes.shape
    modes_flat = modes.reshape(N, Y * X)
    orth_modes_flat = jnp.zeros_like(modes_flat, dtype=modes_flat.dtype)

    for i in range(N):
        v = modes_flat[i]
        for j in range(i):
            v -= jnp.dot(jnp.conj(orth_modes_flat[j]), v) * orth_modes_flat[j]
        v /= jnp.linalg.norm(v) + 1e-12
        orth_modes_flat = orth_modes_flat.at[i].set(v)

    return orth_modes_flat.reshape(N, Y, X)



def orthogonalize_probe_modes_svd(probe_modes, modal_weights, eps=1e-12):
    N, Y, X = probe_modes.shape
    P_flat = probe_modes.reshape(N, Y * X)          # [N, D]

    # SVD: P_flat = U_svd @ diag(s) @ Vh   (Vh rows orthonormal in D-space)
    U_svd, s, Vh = jnp.linalg.svd(P_flat, full_matrices=False)  # U_svd: [N,N], s: [N]

    new_probe_modes = Vh.reshape(N, Y, X)            # already orthonormal rows

    # Same field-invariance logic as before, expressed via the SVD factors
    T = (s[:, None] * jnp.conj(U_svd)).T             # [N, N]  (equivalent role to earlier T)
    new_modal_weights = jnp.einsum("kn,pnm->pkm", T, modal_weights)

    return new_probe_modes, new_modal_weights

