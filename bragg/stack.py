"""Rocking-stack forward model: all rocking angles of one scan position from ONE 3D FFT.

Same conventions as `forward.py` (beam grid (u,v,w), object with Bragg carrier removed,
positions = patch corner in voxels, Fourier shift out(x) = patch(x + shift), orthonormal
detector normalisation, output fftshifted). Differences: one call = one scan position =
a whole rocking stack, output (N_theta, Pu, Pv); XYZ refinement `scan_mistakes` (N, 3).

Why a 3D FFT is possible on the orthogonal object grid
------------------------------------------------------
Rocking phase: exp(2 pi i dQ.r) with r = (u, w) and dQ = (dQ_u, dQ_w). Substituting
x = sin(tB) u + cos(tB) w (the sample x coordinate), w = x/cos(tB) - tan(tB) u:

    dQ.r = c_u u + k_x x,   c_u = -b/cos(tB) = O(theta^2),
    k_x = |Q0| (sin th - tan(tB)(1 - cos th)) ~ |Q0| th   (b = |Q0|(cos th - 1)).

So on the sheared grid (u, v, x) rocking is a ramp in x only, and
    A_k(q_u, q_v) = FFT3[G](q_u, q_v, q_x = -k_x(theta_k)),   G(u,v,x) = f(u, v, w = x/cB - tB u).
The sheared grid is never stored: only the exit wave of one patch is sheared (exact Fourier
shear along w, pitch dx = cos(tB) dw), the probe once per update.

Lattice condition ("fft3" mode): rocking angles must be uniform and
    k_x step = 1/(Nx dx)  <=>  dw = 1 / (Nx cos(tB) |Q0| dtheta)       (see `matched_dw`)
with Nx the (padded) number of x slots. "dft" mode has no condition (arbitrary dw and angles,
exact dQ, cost per angle) and is the reference for the fft3 mode.
"""

from typing import Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .forward import edge_taper_uv, fourier_shift_uv
from .geometry import BraggGeometry, delta_Q_beam_frame
from .probe import build_reference_probe


def matched_dw(theta_B, wavelength, Nx, dtheta):
    """Voxel size along w for which an angular step dtheta falls on the q_x lattice of Nx slots."""
    q0 = 2.0 * np.sin(theta_B) / wavelength
    return 1.0 / (Nx * np.cos(theta_B) * q0 * dtheta)


def matched_rocking_step(geometry: BraggGeometry, Nx):
    """Rocking step matched to the geometry's dw (inverse of `matched_dw`)."""
    return 1.0 / (Nx * np.cos(geometry.theta_B) * geometry.q_mag * geometry.dw)


def min_nx(geometry: BraggGeometry, Pu, Pw):
    """Smallest Nx holding the w-extent of a patch after the u-dependent shear."""
    shear = np.tan(geometry.theta_B) * (Pu // 2) * geometry.du / geometry.dw
    return int(np.ceil(Pw + 2 * shear)) + 2


def lattice_indices(geometry: BraggGeometry, thetas, Nx, tol=0.05):
    """FFT indices along x for the given angles; raises if they are off the q_x lattice."""
    th = np.asarray(thetas, float)
    tB = geometry.theta_B
    kx = geometry.q_mag * (np.sin(th) - np.tan(tB) * (1.0 - np.cos(th)))
    dqx = 1.0 / (Nx * np.cos(tB) * geometry.dw)
    n = -kx / dqx
    err = np.abs(n - np.round(n)).max()
    # the step must also be uniform: check against the matched dw directly
    if err > tol:
        step = np.diff(th).mean() if th.size > 1 else float("nan")
        raise ValueError(
            f"rocking angles are off the 3D-FFT lattice by {err:.2f} q_x pixels (tol {tol}); "
            f"for dtheta={step:.3e} rad and Nx={Nx} use dw="
            f"{matched_dw(tB, geometry.wavelength, Nx, step):.4e} m (now {geometry.dw:.4e}), "
            f"or use mode='dft'"
        )
    return np.round(n).astype(int) % Nx


def pad_w_centered(vol, n):
    """Zero-pad axis 2 to length n keeping the centre index n//2 <-> m//2."""
    m = vol.shape[2]
    lo = n // 2 - m // 2
    return jnp.pad(vol, ((0, 0), (0, 0), (lo, n - m - lo)))


def shear_shift_w(vol, tan_tB, du, dw, shift_w=0.0):
    """out(u, v, w) = vol(u, v, w - tan_tB*u + shift_w*dw): exact Fourier shear along w
    (u-dependent shift) fused with a sub-voxel w shift (shift_w in voxels, same sign
    convention as `fourier_shift_uv`: out(x) = in(x + shift))."""
    Nu, _, Nx = vol.shape
    u = (jnp.arange(Nu) - Nu // 2) * du
    fw = jnp.fft.fftfreq(Nx, d=dw)
    s = tan_tB * u - shift_w * dw  # shift of the content in metres, per u line
    ramp = jnp.exp(-2j * jnp.pi * fw[None, :] * s[:, None])
    return jnp.fft.ifft(jnp.fft.fft(vol, axis=2) * ramp[:, None, :], axis=2)


def make_bragg_stack_model(
    geometry: BraggGeometry,
    patch_shape: Tuple[int, int, int],
    thetas: Sequence[float],
    nx: int = None,
    mode: str = "fft3",
    crop: int = 4,
    max_correction=5.0,
    pad_probe: Tuple[int, int] = (8, 8),
) -> Dict[str, Callable]:
    """Build probe constructor / point model / batch model for rocking stacks.

    thetas: rocking angles (rad, relative to theta_B) shared by every stack (static).
    nx: padded size along the sheared x axis (default `min_nx`).
    mode: "fft3" one 3D FFT (needs the lattice condition) or "dft" (exact dQ, any angles).
    optimizable_params: "object", "probe_2d", "scan_mistakes" (N, 3) -> (u, v, w) in units of
    max_correction voxels via tanh. fixed_params: "positions" (N, 3) patch corner in voxels.
    Output of point_model: (N_theta, Pu, Pv); batch_model: (B, N_theta, Pu, Pv).
    """
    Pu, Pv, Pw = patch_shape
    thetas = np.asarray(thetas, float)
    Nx = int(nx) if nx is not None else min_nx(geometry, Pu, Pw)
    if Nx < min_nx(geometry, Pu, Pw) - 2:
        raise ValueError(f"nx={Nx} too small for the sheared patch (need >= {min_nx(geometry, Pu, Pw)})")
    tB = float(np.tan(geometry.theta_B))
    cB = float(np.cos(geometry.theta_B))
    du, dw = geometry.du, geometry.dw
    idx = lattice_indices(geometry, thetas, Nx) if mode == "fft3" else None
    if mode not in ("fft3", "dft"):
        raise ValueError("mode must be 'fft3' or 'dft'")
    max_corr = jnp.asarray(max_correction, float) * jnp.ones(3)

    def probe_constructor(optimizable_params, fixed_params=None):
        P = build_reference_probe(
            optimizable_params["probe_2d"], geometry, Pw, pad_probe[0], pad_probe[1]
        )
        return shear_shift_w(pad_w_centered(P, Nx), tB, du, dw)

    def select_patch(obj, pos, mistake):
        corner_f = jax.lax.stop_gradient(jnp.floor(pos[:2]))
        cw = jnp.round(jax.lax.stop_gradient(pos[2]))
        res = jnp.concatenate([pos[:2] - corner_f, (pos[2] - cw)[None]])
        corner = corner_f.astype(jnp.int32) - crop
        patch = jax.lax.dynamic_slice(
            obj, (corner[0], corner[1], cw.astype(jnp.int32)), (Pu + 2 * crop, Pv + 2 * crop, Pw)
        )
        shift = res + jnp.tanh(mistake) * max_corr
        patch = fourier_shift_uv(edge_taper_uv(patch, crop), shift[:2])
        patch = patch[crop : crop + Pu, crop : crop + Pv, :]
        return shear_shift_w(pad_w_centered(patch, Nx), tB, du, dw, shift[2])

    def stack_amplitude(exit_volume):
        if mode == "fft3":
            F = jnp.fft.fftn(jnp.fft.ifftshift(exit_volume)) / jnp.sqrt(Pu * Pv)
            sel = jnp.moveaxis(F[:, :, idx], 2, 0)
            return jnp.fft.fftshift(sel, axes=(1, 2))

        u = (jnp.arange(Pu) - Pu // 2) * du
        v = (jnp.arange(Pv) - Pv // 2) * geometry.dv
        x = (jnp.arange(Nx) - Nx // 2) * cB * dw

        def one(theta):
            dQ = delta_Q_beam_frame(theta, geometry.theta_B, geometry.wavelength)
            e_u = jnp.exp(2j * jnp.pi * (dQ[0] - tB * dQ[2]) * u)
            e_v = jnp.exp(2j * jnp.pi * dQ[1] * v)
            e_x = jnp.exp(2j * jnp.pi * (dQ[2] / cB) * x)
            proj = jnp.einsum("uvx,x->uv", exit_volume, e_x) * e_u[:, None] * e_v[None, :]
            return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(proj), norm="ortho"))

        return jax.vmap(one)(jnp.asarray(thetas))

    def point_model_from_probe(P_s, optimizable_params, fixed_params, pos_index):
        patch = select_patch(
            optimizable_params["object"],
            fixed_params["positions"][pos_index],
            optimizable_params["scan_mistakes"][pos_index],
        )
        return jnp.abs(stack_amplitude(P_s * patch)) ** 2

    def point_model(optimizable_params, fixed_params, pos_index):
        P_s = probe_constructor(optimizable_params, fixed_params)
        return point_model_from_probe(P_s, optimizable_params, fixed_params, pos_index)

    def batch_model(optimizable_params, fixed_params, pos_indices):
        P_s = probe_constructor(optimizable_params, fixed_params)  # once per call
        return jax.vmap(
            lambda i: point_model_from_probe(P_s, optimizable_params, fixed_params, i)
        )(pos_indices)

    return {
        "probe_constructor": probe_constructor,
        "select_patch": select_patch,
        "point_model": point_model,
        "batch_model": batch_model,
        "nx": Nx,
        "thetas": thetas,
    }
