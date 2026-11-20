import jax.numpy as jnp
import numpy as np
import pytest

from bragg.probe import (
    angular_spectrum_stack,
    build_reference_probe,
    rotate_volume_fourier_shear,
    shear_via_fourier,
)
from bragg.reference import gaussian_beam_envelope, grid_axes

from conftest import PATCH, W0


def test_shear_is_shift():
    n, d = 64, 1.0
    x = (jnp.arange(n) - n // 2) * d
    f = jnp.exp(-(x[:, None] ** 2) / 20.0) * jnp.ones((1, 8))  # (shear, dep)
    vol = f[:, :, None].astype(complex)
    xd = (jnp.arange(8) - 4) * 1.0
    out = shear_via_fourier(vol, 0, 1, 0.37, d, 1.0)[..., 0]
    expected = jnp.exp(-((x[:, None] - 0.37 * xd[None, :]) ** 2) / 20.0)
    assert jnp.allclose(out, expected, atol=1e-10)


@pytest.mark.parametrize("phi", [0.0, 0.3, -0.7, np.pi / 2])
def test_rotation_vs_analytic_gaussian(phi):
    shape, du, dv, dw = (64, 4, 64), 1.0, 1.0, 1.0
    u, v, w = grid_axes(shape, (du, dv, dw))
    a, b = 3.0, 6.0

    def f(up, wp):
        return jnp.exp(-(up**2) / (2 * a**2) - wp**2 / (2 * b**2))

    nat = jnp.broadcast_to(f(u[:, None, None], w[None, None, :]), shape).astype(complex)
    out = rotate_volume_fourier_shear(nat, phi, du, dw, pad_u=32, pad_w=32)
    c, s = np.cos(phi), np.sin(phi)
    U, W = u[:, None, None], w[None, None, :]
    expected = jnp.broadcast_to(f(c * U - s * W, s * U + c * W), shape)
    assert jnp.allclose(out, expected, atol=2e-4)


def test_rotation_roundtrip_and_energy():
    shape = (64, 4, 64)
    u, v, w = grid_axes(shape, (1.0, 1.0, 1.0))
    vol = jnp.exp(-(u[:, None, None] ** 2 + (w[None, None, :] - 3) ** 2) / 50.0).astype(complex)
    vol = jnp.broadcast_to(vol, shape)
    rot = rotate_volume_fourier_shear(vol, 0.5, 1.0, 1.0, 32, 32)
    back = rotate_volume_fourier_shear(rot, -0.5, 1.0, 1.0, 32, 32)
    assert jnp.allclose(back, vol, atol=1e-6)
    assert jnp.isclose(jnp.sum(jnp.abs(rot) ** 2), jnp.sum(jnp.abs(vol) ** 2), rtol=1e-6)


def test_propagation_matches_gaussian_beam(geom):
    Nu, Nv, Nw = PATCH
    u, v, _ = grid_axes(PATCH, (geom.du, geom.dv, geom.dw))
    U, V = u[:, None], v[None, :]
    f0 = jnp.exp(-(U**2 + V**2) / W0**2).astype(complex)
    z = (np.arange(Nw) - Nw // 2) * geom.dw
    stack = angular_spectrum_stack(f0, geom.wavelength, geom.du, geom.dv, z)
    expected = gaussian_beam_envelope(
        U[..., None], V[..., None], z[None, None, :], W0, geom.wavelength
    )
    assert jnp.max(jnp.abs(stack - expected)) < 1e-4
    assert jnp.allclose(stack[..., Nw // 2], f0)


def test_reference_probe_matches_rotated_analytic_beam(geom):
    Nu, Nv, Nw = PATCH
    u, v, w = grid_axes(PATCH, (geom.du, geom.dv, geom.dw))
    f0 = jnp.exp(-(u[:, None] ** 2 + v[None, :] ** 2) / W0**2).astype(complex)
    P = build_reference_probe(f0, geom, Nw, pad_u=16, pad_w=60, extra_w=80)
    c, s = np.cos(geom.scattering_angle), np.sin(geom.scattering_angle)
    U, V, W = u[:, None, None], v[None, :, None], w[None, None, :]
    expected = gaussian_beam_envelope(c * U - s * W, V, s * U + c * W, W0, geom.wavelength)
    assert jnp.max(jnp.abs(P - expected)) < 3e-3
