"""Rocking-stack model (one 3D FFT per scan position) vs exact 3D reference and vs model 1."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from bragg.forward import make_bragg_model
from bragg.geometry import BraggGeometry
from bragg.reference import exact_intensity_3d, grid_axes, sample_phantom
from bragg.stack import (
    lattice_indices,
    make_bragg_stack_model,
    matched_dw,
    matched_rocking_step,
    min_nx,
)
from loss_and_reg import gauss_loss, join_loss_and_forward_batched
from optimizers import prepare_optimizer

LAM, D, NX, DTH = 1e-10, 30e-9, 64, 1e-4
TB = np.deg2rad(20.0)
DW = matched_dw(TB, LAM, NX, DTH)
PATCH, OBJ = (64, 64, 32), (128, 128, 32)
PITCH = (D, D, DW)
THETAS = (np.arange(11) - 5) * DTH
CUBE = dict(half_size=10 * 30e-9, softness=2 * 30e-9)
POS = np.array([[32.0, 32.0, 0.0], [33.37, 31.81, 0.4], [30.2, 34.6, -0.3]])


@pytest.fixture(scope="module")
def setup():
    g = BraggGeometry(LAM, TB, D, D, DW)
    u, v, _ = grid_axes(PATCH[:2] + (1,), PITCH)
    f0 = jnp.exp(-(u[:, None] ** 2 + v[None, :] ** 2) / (150e-9) ** 2).astype(complex)
    obj = sample_phantom(OBJ, PITCH, **CUBE).astype(complex)
    n = len(POS)
    opt = {"object": obj, "probe_2d": f0, "scan_mistakes": jnp.zeros((n, 3))}
    fx = {"positions": jnp.asarray(POS)}
    return dict(
        g=g, opt=opt, fx=fx,
        fft3=make_bragg_stack_model(g, PATCH, THETAS, nx=NX, mode="fft3"),
        dft=make_bragg_stack_model(g, PATCH, THETAS, nx=NX, mode="dft"),
        m1=make_bragg_model(g, PATCH),
    )


def rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(b))


def test_helpers_consistent(setup):
    g = setup["g"]
    assert np.isclose(matched_rocking_step(g, NX), DTH)
    assert min_nx(g, PATCH[0], PATCH[2]) <= NX


def test_off_lattice_raises():
    bad = BraggGeometry(LAM, TB, D, D, DW * 1.05)
    with pytest.raises(ValueError, match="lattice"):
        lattice_indices(bad, THETAS, NX)
    make_bragg_stack_model(bad, PATCH, THETAS, nx=NX, mode="dft")  # dft has no condition


def test_nx_too_small_raises(setup):
    with pytest.raises(ValueError, match="too small"):
        make_bragg_stack_model(setup["g"], PATCH, THETAS, nx=40, mode="dft")


def test_output_shape(setup):
    out = setup["fft3"]["batch_model"](setup["opt"], setup["fx"], jnp.arange(3))
    assert out.shape == (3, len(THETAS), PATCH[0], PATCH[1])


@pytest.mark.parametrize("mode", ["fft3", "dft"])
@pytest.mark.parametrize("i", [0, 1, 2])
def test_stack_matches_exact_3d_with_xyz_offsets(setup, mode, i):
    """Includes fractional u, v and w offsets (positions 1 and 2)."""
    s = setup
    out = s[mode]["point_model"](s["opt"], s["fx"], i)
    P_ref = s["m1"]["probe_constructor"](dict(s["opt"], scan_mistakes=jnp.zeros((3, 2))))
    p = POS[i]
    patch = sample_phantom(PATCH, PITCH, origin_voxel=(p[0] - 32, p[1] - 32, p[2]), **CUBE)
    ex = jnp.stack([exact_intensity_3d(P_ref, patch, t, s["g"]) for t in THETAS])
    assert rel(out, ex) < 2e-3


def test_fft3_matches_dft_mode(setup):
    s = setup
    a = s["fft3"]["batch_model"](s["opt"], s["fx"], jnp.arange(3))
    b = s["dft"]["batch_model"](s["opt"], s["fx"], jnp.arange(3))
    assert rel(a, b) < 5e-4


def test_matches_model1_per_angle(setup):
    s = setup
    opt1 = dict(s["opt"], scan_mistakes=jnp.zeros((3, 2)))
    stack = s["fft3"]["point_model"](s["opt"], s["fx"], 0)
    for k, t in enumerate(THETAS):
        I1 = s["m1"]["point_model"](
            opt1, {"positions": s["fx"]["positions"], "thetas": jnp.full(3, t)}, 0
        )
        assert float(jnp.max(jnp.abs(stack[k] - I1)) / jnp.max(stack)) < 5e-4


def test_xyz_correction_equals_true_offset(setup):
    """A tanh-bounded correction in (u, v, w) reproduces the same shift of the position."""
    s = setup
    shift = np.array([0.6, -0.4, 0.3])
    pos = jnp.asarray([POS[0]])
    base = {"positions": pos}
    opt = dict(s["opt"], scan_mistakes=jnp.asarray([np.arctanh(shift / 5.0)]))
    corrected = s["fft3"]["point_model"](opt, base, 0)
    moved = s["fft3"]["point_model"](
        dict(s["opt"], scan_mistakes=jnp.zeros((1, 3))), {"positions": pos + shift}, 0
    )
    assert rel(corrected, moved) < 1e-6


def test_w_shift_changes_stack(setup):
    """Sub-voxel w refinement acts (the object here is only Pw deep, so use the correction)."""
    s = setup
    pos = {"positions": jnp.asarray([[32.0, 32.0, 0.0]])}
    a = s["fft3"]["point_model"](dict(s["opt"], scan_mistakes=jnp.zeros((1, 3))), pos, 0)
    m = jnp.asarray([[0.0, 0.0, np.arctanh(0.5 / 5.0)]])
    b = s["fft3"]["point_model"](dict(s["opt"], scan_mistakes=m), pos, 0)
    assert rel(a, b) > 1e-3


def test_jit_and_gradients(setup):
    s = setup
    idx = jnp.arange(3)
    rng = np.random.default_rng(1)
    opt = dict(s["opt"], scan_mistakes=jnp.asarray(0.05 * rng.normal(size=(3, 3))))
    target = s["fft3"]["batch_model"](s["opt"], s["fx"], idx)

    def loss(o):
        return jnp.mean((jnp.sqrt(s["fft3"]["batch_model"](o, s["fx"], idx)) - jnp.sqrt(target)) ** 2)

    val, grads = jax.jit(jax.value_and_grad(loss))(opt)
    assert jnp.isfinite(val)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads.values())
    # all three position components receive gradient
    assert bool(jnp.all(jnp.abs(grads["scan_mistakes"]).max(axis=0) > 0))

    # finite-difference check of the w component
    eps = 1e-5
    e = jnp.zeros((3, 3)).at[0, 2].set(eps)
    fd = (loss(dict(opt, scan_mistakes=opt["scan_mistakes"] + e))
          - loss(dict(opt, scan_mistakes=opt["scan_mistakes"] - e))) / (2 * eps)
    assert abs(float(fd) - float(grads["scan_mistakes"][0, 2])) < 1e-4 * max(1.0, abs(float(fd))) + 1e-9


def test_existing_pipeline_loss_and_optimizer(setup):
    s = setup
    idx = jnp.arange(3)
    measured = s["fft3"]["batch_model"](s["opt"], s["fx"], idx)
    root = jnp.sqrt(measured)
    mask = jnp.ones(measured.shape[1:], bool)
    loss = join_loss_and_forward_batched(gauss_loss, s["fft3"]["point_model"])
    lg = jax.jit(jax.value_and_grad(loss, argnums=0))
    opt = dict(s["opt"], scan_mistakes=0.05 * jnp.ones((3, 3)))
    val, grads = lg(opt, s["fx"], idx, root, mask)
    assert jnp.isfinite(val) and set(grads) == set(opt)
    optimizer = prepare_optimizer({k: {"type": "adam", "learning_rate": 1e-3} for k in opt})
    upd, _ = optimizer.update(grads, optimizer.init(opt), opt)
    new = optax.apply_updates(opt, upd)
    assert lg(new, s["fx"], idx, root, mask)[0] < val
