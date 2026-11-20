"""Beam-grid <-> lab(sample)-grid volume transforms using Fourier shears (no interpolation).

Beam grid: axes (u, v, w), w || k_f at the Bragg condition, (u, v) = detector plane (the grid the
optimisation works on). Lab grid: axes (x, y, z), z || Q0, y = v = rocking axis.
Sample-frame coordinates follow from beam coordinates by (x, z) = (sB u + cB w, -cB u + sB w),
i.e. a rotation about v by pi/2 - theta_B. Both volumes keep the same array shape and pitches
(du, dv, dw) -> (dx, dy, dz); the scalar envelope (Bragg carrier removed) is only resampled,
it is not transformed as a vector. These functions are meant for post-processing / checks
outside of the optimisation loop (they are differentiable, but expensive).

The rotation is split into an exact quarter-turn (index permutation) and a remainder |r| <= 45 deg
done by three exact Fourier shears (Paeth), with zero padding to avoid periodic wrap-around.
"""

import jax.numpy as jnp
import numpy as np

from .probe import rotate_volume_fourier_shear


def _rot90_uw(vol, n_quarter):
    """h(a, b) = f(-b, a) applied n_quarter times on axes (0, 2) (centre index n//2), exact."""
    n = vol.shape[0]
    idx = (2 * (n // 2) - jnp.arange(n)) % n
    for _ in range(int(n_quarter) % 4):
        vol = jnp.swapaxes(jnp.take(vol, idx, axis=0), 0, 2)
    return vol


def rotate_volume_exact_quarter_shear(vol, phi, d_uw, pad=0):
    """Resample g(a,b) = f(a cos(phi) - b sin(phi), a sin(phi) + b cos(phi)) on axes (0, 2).

    Requires equal pitch on both rotating axes and even array sizes there. Output has the
    input shape (the padded box is cropped back).
    """
    Nu, _, Nw = vol.shape
    if Nu % 2 or Nw % 2:
        raise ValueError("axes 0 and 2 must have even length")
    M = max(Nu, Nw) + 2 * pad
    M += M % 2
    lu, lw = (M - Nu) // 2, (M - Nw) // 2
    v = jnp.pad(vol, ((lu, M - Nu - lu), (0, 0), (lw, M - Nw - lw)))
    n = int(np.round(phi / (np.pi / 2)))
    r = phi - n * np.pi / 2
    v = _rot90_uw(v, n)
    if abs(r) > 0:
        v = rotate_volume_fourier_shear(v, r, d_uw, d_uw)
    return v[lu : lu + Nu, :, lw : lw + Nw]


def _check_pitch(du, dw):
    if not np.isclose(du, dw, rtol=1e-9):
        raise ValueError("beam<->lab transform needs du == dw (equal pitch in the rotation plane)")


def beam_to_lab(vol_beam, theta_B, du, dw, pad=32):
    """Object on the beam grid (u,v,w) -> lab grid (x,y,z), z || Q0. Fourier-shear based."""
    _check_pitch(du, dw)
    return rotate_volume_exact_quarter_shear(vol_beam, np.pi / 2 - theta_B, du, pad)


def lab_to_beam(vol_lab, theta_B, du, dw, pad=32):
    """Inverse of `beam_to_lab`: lab grid (x,y,z) -> beam grid (u,v,w)."""
    _check_pitch(du, dw)
    return rotate_volume_exact_quarter_shear(vol_lab, -(np.pi / 2 - theta_B), du, pad)
