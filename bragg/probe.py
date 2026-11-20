"""Probe construction: envelope propagation + differentiable Fourier-shear rotation.

Volumes use axes (u, v, w); physical coordinate of index i along an axis of length n is
(i - n//2) * pitch. The probe is an envelope (carrier exp(2 pi i k_i.r) removed).
"""

import jax.numpy as jnp
import numpy as np


def angular_spectrum_stack(field0, wavelength, du, dv, z_list):
    """Propagate a 2D envelope field (Nu, Nv) to every z in z_list -> (Nu, Nv, Nz).

    Uses the exact scalar transfer function with the carrier removed:
    H = exp(2 pi i (sqrt(1/lam^2 - fu^2 - fv^2) - 1/lam) z), evanescent waves clipped.
    """
    Nu, Nv = field0.shape
    fu = jnp.fft.fftfreq(Nu, d=du)[:, None]
    fv = jnp.fft.fftfreq(Nv, d=dv)[None, :]
    # kz - 1/lam computed as -(fu^2+fv^2)/(kz+1/lam): avoids the catastrophic cancellation of
    # subtracting two ~1e10 1/m numbers, which destroys the phase in float32.
    k0 = 1.0 / wavelength
    f2 = fu**2 + fv**2
    kz = jnp.sqrt(jnp.clip(k0**2 - f2, 0.0))
    phase = 2j * jnp.pi * (-f2 / (kz + k0))
    H = jnp.exp(phase[..., None] * jnp.asarray(z_list)[None, None, :])
    return jnp.fft.ifft2(jnp.fft.fft2(field0)[..., None] * H, axes=(0, 1))


def build_probe_native_volume(field0, wavelength, du, dv, dw, Nw):
    """2D envelope at w' = 0 plane -> native volume (Nu, Nv, Nw), planes at (k - Nw//2)*dw."""
    z_list = (np.arange(Nw) - Nw // 2) * dw
    return angular_spectrum_stack(field0, wavelength, du, dv, z_list)


def shear_via_fourier(vol, shear_axis, dep_axis, factor, d_shear, d_dep):
    """out[x_s, x_d] = vol[x_s - factor * x_d, x_d] via exact per-line Fourier shift."""
    v = jnp.moveaxis(vol, (shear_axis, dep_axis), (0, 1))
    n_s, n_d = v.shape[0], v.shape[1]
    x_d = (jnp.arange(n_d) - n_d // 2) * d_dep
    k_s = jnp.fft.fftfreq(n_s, d=d_shear)
    ramp = jnp.exp(-2j * jnp.pi * k_s[:, None] * (factor * x_d)[None, :])
    ramp = ramp.reshape(ramp.shape + (1,) * (v.ndim - 2))
    v = jnp.fft.ifft(jnp.fft.fft(v, axis=0) * ramp, axis=0)
    return jnp.moveaxis(v, (0, 1), (shear_axis, dep_axis))


def rotate_volume_fourier_shear(vol, phi, du, dw, pad_u=0, pad_w=0):
    """Resample g(u, w) = f(u cos(phi) - w sin(phi), u sin(phi) + w cos(phi)) on the same grid.

    Three-shear (Paeth) decomposition; axis 1 (v) is untouched. Zero padding along u and w
    prevents circular wrap-around; the result is cropped back to the input shape.
    """
    Nu, _, Nw = vol.shape
    v = jnp.pad(vol, ((pad_u, pad_u), (0, 0), (pad_w, pad_w)))
    t = jnp.tan(phi / 2.0)
    s = jnp.sin(phi)
    v = shear_via_fourier(v, 0, 2, t, du, dw)
    v = shear_via_fourier(v, 2, 0, -s, dw, du)
    v = shear_via_fourier(v, 0, 2, t, du, dw)
    return v[pad_u : pad_u + Nu, :, pad_w : pad_w + Nw]


def build_reference_probe(field0, geometry, Nw, pad_u=8, pad_w=8, extra_w=None):
    """2D envelope -> propagated native volume -> rotated by 2 theta_B into the beam frame.

    The rotated native box does not cover the reference box, so the native volume is
    propagated over Nw + 2*extra_w planes (default ceil(Nu*|sin(2 tB)|) per side; truncation error of the diverging beam decays with larger extra_w) and the
    result is cropped back to the central Nw planes.
    """
    Nu = field0.shape[0]
    phi = geometry.scattering_angle
    if extra_w is None:
        extra_w = int(np.ceil(Nu * abs(np.sin(phi))))
    native = build_probe_native_volume(
        field0, geometry.wavelength, geometry.du, geometry.dv, geometry.dw, Nw + 2 * extra_w
    )
    rot = rotate_volume_fourier_shear(
        native, phi, geometry.du, geometry.dw, pad_u, pad_w
    )
    return rot[:, :, extra_w : extra_w + Nw]
