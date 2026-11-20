import jax.numpy as jnp
import numpy as np
import pytest

from bragg.frames import beam_to_lab, lab_to_beam
from bragg.reference import lab_function_on_beam_grid, lab_function_on_lab_grid

D = 30e-9
SHAPE = (96, 24, 96)
PITCH = (D, D, D)


def particle(x, y, z):
    """Smooth lab-frame particle (elongated along z) with slow phase."""
    r2 = (x / (8 * D)) ** 2 + (y / (3 * D)) ** 2 + (z / (9 * D)) ** 2
    return jnp.exp(-r2) * jnp.exp(1j * (1.2 * z / (9 * D) + 0.7 * x * y / (8 * D * 3 * D)))


@pytest.mark.parametrize("deg", [0.0, 12.0, 20.0, 35.0, 60.0])
def test_beam_to_lab_matches_analytic(deg):
    tB = np.deg2rad(deg)
    beam = lab_function_on_beam_grid(particle, SHAPE, PITCH, tB)
    lab = lab_function_on_lab_grid(particle, SHAPE, PITCH)
    out = beam_to_lab(beam, tB, D, D, pad=48)
    assert jnp.max(jnp.abs(out - lab)) < 2e-4


@pytest.mark.parametrize("deg", [12.0, 20.0, 35.0])
def test_lab_to_beam_matches_analytic(deg):
    tB = np.deg2rad(deg)
    beam = lab_function_on_beam_grid(particle, SHAPE, PITCH, tB)
    lab = lab_function_on_lab_grid(particle, SHAPE, PITCH)
    out = lab_to_beam(lab, tB, D, D, pad=48)
    assert jnp.max(jnp.abs(out - beam)) < 2e-4


def test_roundtrip_and_energy():
    tB = np.deg2rad(20.0)
    beam = lab_function_on_beam_grid(particle, SHAPE, PITCH, tB)
    lab = beam_to_lab(beam, tB, D, D, pad=48)
    back = lab_to_beam(lab, tB, D, D, pad=48)
    assert jnp.max(jnp.abs(back - beam)) < 2e-4
    assert jnp.isclose(jnp.sum(jnp.abs(lab) ** 2), jnp.sum(jnp.abs(beam) ** 2), rtol=1e-3)


def test_gradient_flows():
    import jax

    tB = np.deg2rad(20.0)
    beam = lab_function_on_beam_grid(particle, SHAPE, PITCH, tB)
    g = jax.grad(lambda b: jnp.sum(jnp.abs(beam_to_lab(b, tB, D, D, pad=16)) ** 2))(beam)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_unequal_pitch_raises():
    with pytest.raises(ValueError):
        beam_to_lab(jnp.zeros(SHAPE, complex), 0.3, D, 2 * D)
