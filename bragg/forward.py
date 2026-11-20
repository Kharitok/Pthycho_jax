"""Differentiable 3D Bragg ptychography forward model (rocking-angle phase-ramp model).

Per scan point (vmapped):
  object patch (integer corner, sub-voxel Fourier shift in u,v incl. trainable correction)
  x reference probe volume (built once per update) -> exit volume
  x exp(+2 pi i dQ(theta).r) -> sum over w -> 2D FFT -> |.|^2

Same structure/signatures as the transmission models: point_model(optimizable_params,
fixed_params, pos_index). optimizable_params: "object" (Ou,Ov,Ow complex), "probe_2d"
(Pu,Pv complex), "scan_mistakes" (N,2). fixed_params: "positions" (N,3) voxel units of the
patch origin in object coordinates (u,v,w), "thetas" (N,) rocking angles in rad.
Detector amplitude uses an orthonormal FFT of sum_w(...) (no dw factor).
"""

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from .geometry import BraggGeometry, delta_Q_beam_frame
from .probe import build_reference_probe


def edge_taper_uv(patch, taper_pixels):
    """Smooth sin edge window on axes (u, v) of a (Pu, Pv, Pw) patch."""
    if taper_pixels == 0:
        return patch
    ramp = jnp.sin(jnp.linspace(0, jnp.pi / 2, taper_pixels + 2)[1:-1])

    def win(n):
        return jnp.concatenate([ramp, jnp.ones(n - 2 * taper_pixels), ramp[::-1]])

    return patch * win(patch.shape[0])[:, None, None] * win(patch.shape[1])[None, :, None]


def fourier_shift_uv(patch, shift):
    """out(x) = patch(x + shift) along (u, v), shift in voxels (periodic, exact for bandlimited)."""
    fu = jnp.fft.fftfreq(patch.shape[0])[:, None, None]
    fv = jnp.fft.fftfreq(patch.shape[1])[None, :, None]
    ramp = jnp.exp(2j * jnp.pi * (fu * shift[0] + fv * shift[1]))
    return jnp.fft.ifft2(jnp.fft.fft2(patch, axes=(0, 1)) * ramp, axes=(0, 1))


def rocking_ramps_1d(dQ, shape, geometry: BraggGeometry):
    """Separable ramp factors exp(+2 pi i dQ_a x_a) for a=u,v,w (patch-centred coordinates)."""
    out = []
    for n, d, q in zip(shape, (geometry.du, geometry.dv, geometry.dw), dQ):
        x = (jnp.arange(n) - n // 2) * d
        out.append(jnp.exp(2j * jnp.pi * q * x))
    return out


def detector_amplitude(exit_volume, dQ, geometry: BraggGeometry):
    e_u, e_v, e_w = rocking_ramps_1d(dQ, exit_volume.shape, geometry)
    proj = jnp.einsum("uvw,w->uv", exit_volume, e_w) * e_u[:, None] * e_v[None, :]
    field = jnp.fft.fft2(jnp.fft.ifftshift(proj), norm="ortho")
    return jnp.fft.fftshift(field)


def make_bragg_model(
    geometry: BraggGeometry,
    patch_shape: Tuple[int, int, int],
    crop: int = 4,
    max_correction: float = 5.0,
    pad_probe: Tuple[int, int] = (8, 8),
) -> Dict[str, Callable]:
    """Build probe constructor, point model and batch model.

    crop pixels (u,v) are cut from each side of the shifted patch (and tapered) to avoid
    Fourier-shift wrap-around; positions must keep corner - crop inside the object.
    """
    Pu, Pv, Pw = patch_shape

    def probe_constructor(optimizable_params, fixed_params=None):
        return build_reference_probe(
            optimizable_params["probe_2d"], geometry, Pw, pad_probe[0], pad_probe[1]
        )

    def select_patch(obj, pos, mistake):
        corner_f = jax.lax.stop_gradient(jnp.floor(pos[:2]))
        residual = pos[:2] - corner_f
        corner = corner_f.astype(jnp.int32) - crop
        cw = jnp.round(jax.lax.stop_gradient(pos[2])).astype(jnp.int32)
        patch = jax.lax.dynamic_slice(
            obj, (corner[0], corner[1], cw), (Pu + 2 * crop, Pv + 2 * crop, Pw)
        )
        shift = residual + jnp.tanh(mistake) * max_correction
        patch = fourier_shift_uv(edge_taper_uv(patch, crop), shift)
        return patch[crop : crop + Pu, crop : crop + Pv, :]

    def point_model_from_probe(P_ref, optimizable_params, fixed_params, pos_index):
        patch = select_patch(
            optimizable_params["object"],
            fixed_params["positions"][pos_index],
            optimizable_params["scan_mistakes"][pos_index],
        )
        dQ = delta_Q_beam_frame(
            fixed_params["thetas"][pos_index], geometry.theta_B, geometry.wavelength
        )
        amp = detector_amplitude(P_ref * patch, dQ, geometry)
        return jnp.abs(amp) ** 2

    def point_model(optimizable_params, fixed_params, pos_index):
        P_ref = probe_constructor(optimizable_params, fixed_params)
        return point_model_from_probe(P_ref, optimizable_params, fixed_params, pos_index)

    def batch_model(optimizable_params, fixed_params, pos_indices):
        P_ref = probe_constructor(optimizable_params, fixed_params)  # once per call
        return jax.vmap(
            lambda i: point_model_from_probe(P_ref, optimizable_params, fixed_params, i)
        )(pos_indices)

    return {
        "probe_constructor": probe_constructor,
        "select_patch": select_patch,
        "point_model": point_model,
        "batch_model": batch_model,
    }
