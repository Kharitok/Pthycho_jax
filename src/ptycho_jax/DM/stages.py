"""Exact stages of the existing tapered-object Fourier-shift forward model."""

import jax
import jax.numpy as jnp
from .conventions import fft2c, ifft2c


def taper(shape, margin, dtype):
    """Real cosine edge taper (Y+2m,X+2m), identical to the AD selector."""
    edge = jnp.sin(jnp.linspace(0, jnp.pi / 2, margin, dtype=dtype))
    wy = jnp.concatenate((edge, jnp.ones(shape[0] - 2 * margin, dtype), edge[::-1]))
    wx = jnp.concatenate((edge, jnp.ones(shape[1] - 2 * margin, dtype), edge[::-1]))
    return wy[:, None] * wx[None, :]


def shift_ramps(shifts, shape, dtype):
    """Fourier ramps for translations (J,2) -> (J,Y,X)."""
    shifts = shifts.astype(dtype)
    fy = jnp.fft.fftfreq(shape[0], dtype=dtype)
    fx = jnp.fft.fftfreq(shape[1], dtype=dtype)
    phase = (
        shifts[:, 0, None, None] * fy[None, :, None]
        + shifts[:, 1, None, None] * fx[None, None, :]
    )
    return jnp.exp(jnp.asarray(-2j * jnp.pi, dtype=jnp.result_type(dtype, 1j)) * phase)


def shift_patches(patches, shifts):
    """Circular Fourier translations of complex patches (J,Y,X), shifts (J,2)."""
    return ifft2c(
        fft2c(patches) * shift_ramps(shifts, patches.shape[-2:], patches.real.dtype)
    )


def extract_patches(obj, geom):
    """Integer enlarged extraction E: canvas (H,W) -> (J,Y+2m,X+2m)."""
    shape = tuple(s + 2 * geom.margin for s in geom.frame_shape)
    return jax.vmap(lambda n: jax.lax.dynamic_slice(obj, n, shape))(geom.n)


def object_patches(obj, geom):
    """Exact L_j O: canvas (H,W) -> tapered, shifted, cropped (J,Y,X)."""
    m = geom.margin
    if geom.integer_only:
        return jax.vmap(lambda n: jax.lax.dynamic_slice(obj, n + m, geom.frame_shape))(
            geom.n
        )
    patches = extract_patches(obj, geom)
    patches = patches * taper(patches.shape[-2:], m, obj.real.dtype)
    shifted = shift_patches(patches, -geom.delta)
    return shifted[:, m:-m, m:-m]


def exit_waves(probe, obj, a, geom):
    """Probe (K,Y,X), canvas (H,W), amplitudes (J,K) -> (J,K,Y,X)."""
    return a[:, :, None, None] * probe[None] * object_patches(obj, geom)[:, None]


def detector_intensity(fpsi):
    """Complex detector waves (J,K,Y,X) -> incoherent intensities (J,Y,X)."""
    return jnp.sum(fpsi.real**2 + fpsi.imag**2, axis=1)


def forward_from_stages(probe, obj, a, geom):
    """Reference-equivalent intensity (J,Y,X) for fixed integer/fractional positions."""
    return detector_intensity(fft2c(exit_waves(probe, obj, a, geom)))
