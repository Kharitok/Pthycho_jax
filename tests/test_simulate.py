"""Simulator (direct 3D FFT) vs the reconstruction models, and reconstruction smoke test."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bragg.recon_utils import corner_positions
from bragg.simulate import ExperimentConfig, SimulatedData, simulate_dataset
from bragg.stack import make_bragg_stack_model


@pytest.fixture(scope="module")
def sim():
    cfg = ExperimentConfig(n_det=64, patch_w=32, obj_margin=24, n_rock=5, n_scan_x=3, n_scan_y=2,
                           photons_max=None)
    return simulate_dataset(cfg)


def _model(sim, mode="dft"):
    c = sim.cfg
    return make_bragg_stack_model(sim.geom, (c.n_det, c.n_det, c.patch_w), sim.thetas, nx=sim.nx,
                                  mode=mode, crop=4, max_correction=(50.0, 50.0, 50.0))


def _predict(sim, sign):
    te = sim.true_vox - sim.nominal_vox
    p = {"object": jnp.asarray(sim.true_object), "probe_2d": jnp.asarray(sim.probe_2d),
         "scan_mistakes": jnp.arctanh(jnp.asarray(sign * te) / 50.0)}
    fx = {"positions": jnp.asarray(corner_positions(sim))}
    return np.asarray(_model(sim)["batch_model"](p, fx, jnp.arange(len(te)))), p, fx


def test_model_matches_direct_simulation_with_true_positions(sim):
    out, _, _ = _predict(sim, +1)
    err = [np.linalg.norm(o - r) / np.linalg.norm(r) for o, r in zip(out, sim.clean)]
    assert max(err) < 3e-2


def test_wrong_position_sign_is_worse(sim):
    good, _, _ = _predict(sim, +1)
    bad, _, _ = _predict(sim, -1)
    eg = np.linalg.norm(good - sim.clean)
    assert np.linalg.norm(bad - sim.clean) > 3 * eg


def test_save_load_roundtrip(sim, tmp_path):
    sim.save(tmp_path / "s.npz")
    s2 = SimulatedData.load(tmp_path / "s.npz")
    assert np.allclose(s2.intensities, sim.intensities)
    assert s2.cfg.n_det == sim.cfg.n_det and s2.cfg.photons_max == sim.cfg.photons_max
    assert s2.nx == sim.nx


def test_gradient_descent_reduces_loss(sim):
    from loss_and_reg import gauss_loss, join_loss_and_forward_batched
    from bragg.recon_utils import initial_object, initial_probe_2d

    m = _model(sim)
    g = sim.geom
    params = {"object": jnp.asarray(initial_object(sim.obj_shape, (g.du, g.dv, g.dw), 0.3e-6)),
              "probe_2d": jnp.asarray(initial_probe_2d(sim, 1.25)),
              "scan_mistakes": jnp.zeros((sim.intensities.shape[0], 3))}
    fx = {"positions": jnp.asarray(corner_positions(sim))}
    root = jnp.asarray(np.sqrt(sim.intensities))
    mask = jnp.ones(sim.intensities.shape[-2:], bool)
    loss = join_loss_and_forward_batched(gauss_loss, m["point_model"])
    vg = jax.jit(jax.value_and_grad(loss, argnums=0))
    idx = jnp.arange(root.shape[0])
    l0, gr = vg(params, fx, idx, root, mask)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in gr.values())
    p2 = {k: params[k] - 1e-3 * gr[k] / (jnp.abs(gr[k]).max() + 1e-30) for k in params}
    assert float(loss(p2, fx, idx, root, mask)) < float(l0)
