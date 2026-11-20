"""Bragg geometry primitives for the 3D Bragg ptychography forward model.

Frames (all right handed, lengths in m, reciprocal lengths in 1/m, no 2*pi):
  sample frame (x, y, z): z || Q0 (Bragg vector at theta = 0), rocking axis = y.
  beam frame   (u, v, w): w || k_f, v = y, u = v x w. The detector plane is (u, v).
  native frame (u', v, w'): w' || k_i (probe propagation axis), u' = v x w'.

With k_f = (cos tB, 0, sin tB)/lam and k_i = (cos tB, 0, -sin tB)/lam in the sample frame:
  u = ( sin tB, 0, -cos tB),  w = k_f/|k_f|
  w' = (sin 2tB, cos 2tB),    u' = (cos 2tB, -sin 2tB)   (components along (u, w)).

The rocking angle theta is the sample rotation about y relative to the Bragg
condition (theta = 0): Q(theta) = R_y(theta) Q0 with
R_y(theta)(x, z) = (x cos + z sin, -x sin + z cos).

Fourier convention: F(q) = sum_r f(r) exp(-2 pi i q.r).
The object is stored with the Bragg carrier exp(2 pi i Q0.r) removed, so rocking by
theta multiplies the exit volume by exp(+2 pi i dQ(theta).r), dQ = Q(theta) - Q0.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class BraggGeometry:
    wavelength: float  # m
    theta_B: float  # rad
    du: float  # m, voxel pitch along u (= lam * D / (N_det * p))
    dv: float  # m
    dw: float  # m

    @property
    def q_mag(self):
        return 2.0 * np.sin(self.theta_B) / self.wavelength

    @property
    def scattering_angle(self):
        return 2.0 * self.theta_B


def rotation_matrix_y(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    zero, one = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.stack(
        [
            jnp.stack([c, zero, s]),
            jnp.stack([zero, one, zero]),
            jnp.stack([-s, zero, c]),
        ]
    )


def beam_basis_matrix(theta_B):
    """Rows are (u, v, w) expressed in the sample frame; maps sample -> beam comps."""
    sB, cB = jnp.sin(theta_B), jnp.cos(theta_B)
    zero, one = jnp.zeros_like(sB), jnp.ones_like(sB)
    u = jnp.stack([sB, zero, -cB])
    v = jnp.stack([zero, one, zero])
    w = jnp.stack([cB, zero, sB])
    return jnp.stack([u, v, w])


def bragg_Q_vector(theta, theta_B, wavelength):
    """Exact Q(theta) = R_y(theta) Q0 in the sample frame."""
    q0 = jnp.array([0.0, 0.0, 1.0]) * (2.0 * jnp.sin(theta_B) / wavelength)
    return rotation_matrix_y(theta) @ q0


def delta_Q_beam_frame(theta, theta_B, wavelength):
    """dQ(theta) = Q(theta) - Q(0) in the beam frame (u, v, w). Differentiable in theta.

    Closed form dQ_sample = |Q0| (sin th, 0, cos th - 1) with cos th - 1 = -2 sin^2(th/2):
    subtracting the full vectors (|Q0| ~ 4e9 1/m) loses ~180 1/m in float32.
    """
    theta = jnp.asarray(theta)
    q0 = 2.0 * jnp.sin(theta_B) / wavelength
    zero = jnp.zeros_like(theta)
    dq = q0 * jnp.stack([jnp.sin(theta), zero, -2.0 * jnp.sin(theta / 2.0) ** 2])
    return beam_basis_matrix(theta_B) @ dq


def scan_to_grid_positions(X, Y, theta_B, du, dv, dw, slide_along_beam=True):
    """Beam position on the sample (X along x, Y along y, in m) -> beam-grid voxel offsets.

    A beam passing through X on the sample x axis sits at X*(sin tB, cos tB) along (u, w).
    The beam axis in the beam frame is (sin 2tB, 0, cos 2tB), and the beam is invariant
    along it, so with slide_along_beam=True the point is slid along the axis to w = 0:
    the patch window then stays centred on the particle and u = -X sin(tB)/cos(2tB).
    With slide_along_beam=False the raw (u, w) = X (sin tB, cos tB) is returned.
    Returns (..., 3) in voxel units, order (u, v, w).
    """
    if slide_along_beam:
        pu = -X * jnp.sin(theta_B) / jnp.cos(2.0 * theta_B) / du
        pw = jnp.zeros_like(pu)
    else:
        pu = X * jnp.sin(theta_B) / du
        pw = X * jnp.cos(theta_B) / dw
    pv = Y / dv
    return jnp.stack([pu, pv, pw], axis=-1)
