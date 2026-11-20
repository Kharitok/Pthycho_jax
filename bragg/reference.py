"""Reference (slow, exact) calculations used to validate the fast forward model.

* analytic soft-cube phantom with a slow phase (strain-like) and analytic Gaussian beam
* exact 3D Fourier transform of (probe * object) evaluated by separable matrix DFTs at the
  detector q-points of a given rocking angle
* optional: exact treatment where the object envelope is physically rotated by theta
"""

import jax.numpy as jnp

from .geometry import BraggGeometry, beam_basis_matrix, delta_Q_beam_frame, rotation_matrix_y


def grid_axes(shape, pitches, origin_voxel=(0.0, 0.0, 0.0)):
    return [
        (jnp.arange(n) - n // 2 + o) * d for n, d, o in zip(shape, pitches, origin_voxel)
    ]


def soft_cube_phantom(u, v, w, half_size, softness, phase_coeffs=(1.5, 0.8, 0.6)):
    """Smooth cube (analytic, bandlimited-ish) with slow phase. u, v, w: broadcastable arrays (m)."""
    def edge(x):
        return 0.5 * (jnp.tanh((half_size - jnp.abs(x)) / softness) + 1.0)

    a = half_size
    amp = edge(u) * edge(v) * edge(w)
    c1, c2, c3 = phase_coeffs
    phase = c1 * (u / a) ** 2 + c2 * (v * w) / a**2 + c3 * (w / a) + 0.5 * (u / a)
    return amp * jnp.exp(1j * phase)


def sample_phantom(shape, pitches, origin_voxel=(0.0, 0.0, 0.0), **kw):
    u, v, w = grid_axes(shape, pitches, origin_voxel)
    return soft_cube_phantom(u[:, None, None], v[None, :, None], w[None, None, :], **kw)


def gaussian_beam_envelope(u, v, z, w0, wavelength):
    """Paraxial Gaussian beam envelope (carrier exp(ikz) removed); z along propagation."""
    zR = jnp.pi * w0**2 / wavelength
    q = 1.0 + 1j * z / zR
    return jnp.exp(-(u**2 + v**2) / (w0**2 * q)) / q


def exact_amplitude_3d(f, dQ, geometry: BraggGeometry):
    """A(qu,qv) = (Nu Nv)^-1/2 sum_r f(r) exp(2 pi i dQ.r) exp(-2 pi i (qu u + qv v)).

    f: (Nu,Nv,Nw) exit volume. Evaluated with explicit matrix DFTs (no FFT/projection trick),
    on the same fftshift-ordered q grid as the model output.
    """
    Nu, Nv, Nw = f.shape
    u, v, w = grid_axes(f.shape, (geometry.du, geometry.dv, geometry.dw))
    qu = (jnp.arange(Nu) - Nu // 2) / (Nu * geometry.du)
    qv = (jnp.arange(Nv) - Nv // 2) / (Nv * geometry.dv)
    Eu = jnp.exp(-2j * jnp.pi * (qu[:, None] - dQ[0]) * u[None, :])
    Ev = jnp.exp(-2j * jnp.pi * (qv[:, None] - dQ[1]) * v[None, :])
    Ew = jnp.exp(2j * jnp.pi * dQ[2] * w)  # detector plane has q_w = 0
    return jnp.einsum("ju,kv,w,uvw->jk", Eu, Ev, Ew, f) / jnp.sqrt(Nu * Nv)


def exact_intensity_3d(probe_vol, obj_vol, theta, geometry: BraggGeometry):
    dQ = delta_Q_beam_frame(theta, geometry.theta_B, geometry.wavelength)
    return jnp.abs(exact_amplitude_3d(probe_vol * obj_vol, dQ, geometry)) ** 2


def rotated_phantom(shape, pitches, theta, theta_B, origin_voxel=(0.0, 0.0, 0.0), **kw):
    """Phantom envelope physically rotated by theta (about v) in the beam frame."""
    B = beam_basis_matrix(theta_B)
    R = B @ rotation_matrix_y(theta) @ B.T
    u, v, w = grid_axes(shape, pitches, origin_voxel)
    U, V, W = jnp.meshgrid(u, v, w, indexing="ij")
    r = jnp.stack([U, V, W], axis=-1) @ R  # R^T r for every point -> envelope at R^-1 r
    return soft_cube_phantom(r[..., 0], r[..., 1], r[..., 2], **kw)


def lab_function_on_beam_grid(fn_lab, shape, pitches, theta_B, origin_voxel=(0.0, 0.0, 0.0)):
    """Evaluate a user-defined particle fn_lab(x, y, z) (analytic, lab frame, z || Q0) exactly on the
    beam grid (u,v,w): interpolation-free ground truth for `frames.lab_to_beam`."""
    u, v, w = grid_axes(shape, pitches, origin_voxel)
    U, V, W = jnp.meshgrid(u, v, w, indexing="ij")
    B = beam_basis_matrix(theta_B)
    r = jnp.stack([U, V, W], axis=-1) @ B  # (x, y, z) = B^T (u, v, w)
    return fn_lab(r[..., 0], r[..., 1], r[..., 2])


def lab_function_on_lab_grid(fn_lab, shape, pitches):
    x, y, z = grid_axes(shape, pitches)
    return fn_lab(x[:, None, None], y[None, :, None], z[None, None, :])
