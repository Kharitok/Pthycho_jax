import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bragg.forward as fwd
from bragg.forward import make_bragg_model
from bragg.reference import (
    exact_intensity_3d,
    gaussian_beam_envelope,
    grid_axes,
    rotated_phantom,
    sample_phantom,
    soft_cube_phantom,
)

from conftest import CUBE, OBJ, PATCH, W0

MAX_CORR = 5.0


@pytest.fixture(scope="module")
def setup(geom):
    pitches = (geom.du, geom.dv, geom.dw)
    u, v, _ = grid_axes(PATCH[:2] + (1,), pitches)
    f0 = jnp.exp(-(u[:, None] ** 2 + v[None, :] ** 2) / W0**2).astype(complex)
    model = make_bragg_model(geom, PATCH, crop=4, max_correction=MAX_CORR)
    obj = sample_phantom(OBJ, pitches, **CUBE).astype(complex)
    opt = {"object": obj, "probe_2d": f0, "scan_mistakes": jnp.zeros((6, 2))}
    P_ref = model["probe_constructor"](opt)
    return dict(model=model, obj=obj, opt=opt, P_ref=P_ref, pitches=pitches)


def fixed(positions, thetas):
    return {"positions": jnp.asarray(positions, float), "thetas": jnp.asarray(thetas, float)}


def rel_err(I, ref):
    return float(jnp.max(jnp.abs(I - ref)) / jnp.max(ref))


CENTER = (32.0, 32.0, 0.0)  # patch origin so the cube sits at the patch centre


def test_theta0_equals_3d_fft_plane(setup):
    m, P = setup["model"], setup["P_ref"]
    opt, Nw = setup["opt"], PATCH[2]
    I = m["point_model"](opt, fixed([CENTER], [0.0]), 0)
    f = P * setup["obj"][32:96, 32:96, :]
    F = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(f), norm="ortho"))
    I3 = jnp.abs(F[:, :, Nw // 2]) ** 2 * Nw  # ortho(3D) -> ortho(2D) of the w-sum
    assert rel_err(I, I3) < 1e-10


@pytest.mark.parametrize("theta", [0.0, 3e-4, -6e-4, 1.2e-3, -1.6e-3])
def test_rocking_matches_exact_3d_transform(setup, geom, theta):
    m, P = setup["model"], setup["P_ref"]
    I = m["point_model"](setup["opt"], fixed([CENTER], [theta]), 0)
    truth = exact_intensity_3d(P, setup["obj"][32:96, 32:96, :], theta, geom)
    assert rel_err(I, truth) < 1e-9


def test_fractional_position_matches_analytic_object(setup, geom):
    m, P, pitches = setup["model"], setup["P_ref"], setup["pitches"]
    pos = np.array([33.37, 31.81, 0.0])
    theta = 8e-4
    I = m["point_model"](setup["opt"], fixed([pos], [theta]), 0)
    patch = sample_phantom(
        PATCH, pitches, origin_voxel=(pos[0] - 32, pos[1] - 32, pos[2]), **CUBE
    )
    truth = exact_intensity_3d(P, patch, theta, geom)
    assert rel_err(I, truth) < 2e-3


def test_position_correction_equals_true_offset(setup, geom):
    """Nominal integer position + trainable correction == true sub-voxel position."""
    m, P, pitches = setup["model"], setup["P_ref"], setup["pitches"]
    true_off = np.array([0.6, -0.4])
    opt = dict(setup["opt"])
    mistakes = jnp.zeros((6, 2)).at[0].set(jnp.arctanh(true_off / MAX_CORR))
    opt["scan_mistakes"] = mistakes
    I = m["point_model"](opt, fixed([CENTER], [5e-4]), 0)
    patch = sample_phantom(
        PATCH, pitches, origin_voxel=(true_off[0], true_off[1], 0.0), **CUBE
    )
    truth = exact_intensity_3d(P, patch, 5e-4, geom)
    assert rel_err(I, truth) < 2e-3


def test_w_position_integer_shift(setup, geom):
    m, P, pitches = setup["model"], setup["P_ref"], setup["pitches"]
    obj = sample_phantom((128, 128, 48), pitches, **CUBE).astype(complex)
    opt = dict(setup["opt"], object=obj)
    pos = np.array([32.0, 32.0, 5.0])
    I = m["point_model"](opt, fixed([pos], [0.0]), 0)
    patch = sample_phantom(PATCH, pitches, origin_voxel=(0.0, 0.0, -3.0), **CUBE)
    truth = exact_intensity_3d(P, patch, 0.0, geom)
    assert rel_err(I, truth) < 1e-9


def test_parseval(setup, geom):
    m, P = setup["model"], setup["P_ref"]
    I = m["point_model"](setup["opt"], fixed([CENTER], [0.0]), 0)
    proj = jnp.sum(P * setup["obj"][32:96, 32:96, :], axis=2)
    assert jnp.isclose(jnp.sum(I), jnp.sum(jnp.abs(proj) ** 2), rtol=1e-10)


def test_rotated_envelope_error_grows_with_angle(setup, geom):
    """Documents the validity range of the phase-ramp approximation (envelope not rotated)."""
    m, P, pitches = setup["model"], setup["P_ref"], setup["pitches"]
    errs = []
    thetas = [0.0, 2.5e-4, 5e-4, 1e-3, 1.5e-3]
    for th in thetas:
        I = m["point_model"](setup["opt"], fixed([CENTER], [th]), 0)
        rot = rotated_phantom(PATCH, pitches, th, geom.theta_B, **CUBE)
        errs.append(rel_err(I, exact_intensity_3d(P, rot, th, geom)))
    print("theta[rad] -> rel err:", list(zip(thetas, errs)))
    assert errs[0] < 1e-9
    assert errs[1] < errs[-1]
    assert errs[-1] < 0.1
