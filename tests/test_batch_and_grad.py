import jax
import jax.numpy as jnp
import numpy as np

import bragg.forward as fwd
from bragg.forward import make_bragg_model
from bragg.reference import grid_axes, sample_phantom

from conftest import CUBE, OBJ, PATCH, W0

N = 5


def make(geom, seed=0):
    pitches = (geom.du, geom.dv, geom.dw)
    u, v, _ = grid_axes(PATCH[:2] + (1,), pitches)
    f0 = jnp.exp(-(u[:, None] ** 2 + v[None, :] ** 2) / W0**2).astype(complex)
    model = make_bragg_model(geom, PATCH, crop=4)
    obj = sample_phantom(OBJ, pitches, **CUBE).astype(complex)
    rng = np.random.default_rng(seed)
    opt = {
        "object": obj,
        "probe_2d": f0,
        "scan_mistakes": jnp.asarray(0.1 * rng.normal(size=(N, 2))),
    }
    pos = np.stack(
        [32 + rng.uniform(-6, 6, N), 32 + rng.uniform(-6, 6, N), np.zeros(N)], -1
    )
    fx = {
        "positions": jnp.asarray(pos),
        "thetas": jnp.asarray(rng.uniform(-1e-3, 1e-3, N)),
    }
    return model, opt, fx


def test_vmap_matches_loop(geom):
    model, opt, fx = make(geom)
    idx = jnp.arange(N)
    batch = model["batch_model"](opt, fx, idx)
    loop = jnp.stack([model["point_model"](opt, fx, i) for i in range(N)])
    assert batch.shape == (N,) + PATCH[:2]
    assert jnp.allclose(batch, loop, rtol=1e-9, atol=1e-18)


def test_probe_built_once_per_batch(geom, monkeypatch):
    calls = []
    orig = fwd.build_reference_probe
    monkeypatch.setattr(
        fwd, "build_reference_probe", lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    )
    model, opt, fx = make(geom)
    model["batch_model"](opt, fx, jnp.arange(N))
    assert len(calls) == 1


def test_jit_no_recompile(geom):
    model, opt, fx = make(geom)
    f = jax.jit(model["batch_model"])
    f(opt, fx, jnp.arange(N)).block_until_ready()
    opt2 = dict(opt, scan_mistakes=opt["scan_mistakes"] + 0.01)
    f(opt2, fx, jnp.arange(N)).block_until_ready()
    assert f._cache_size() == 1


def directional_check(loss, params, key, direction, eps=1e-6):
    g = jax.grad(loss)(params)[key]
    dL = jnp.real(jnp.sum(g * direction))
    plus = loss({**params, key: params[key] + eps * direction})
    minus = loss({**params, key: params[key] - eps * direction})
    fd = (plus - minus) / (2 * eps)
    assert jnp.isfinite(dL)
    assert abs(dL - fd) < 1e-5 * max(abs(fd), 1e-30) + 1e-12, (key, dL, fd)


def test_gradients_match_finite_differences(geom):
    model, opt, fx = make(geom)
    idx = jnp.arange(N)
    target = model["batch_model"](
        dict(opt, scan_mistakes=jnp.zeros((N, 2))), fx, idx
    ) * 1.1

    def loss(p):
        I = model["batch_model"](p, fx, idx)
        return jnp.sum((jnp.sqrt(I) - jnp.sqrt(target)) ** 2)

    rng = np.random.default_rng(1)
    for key in ("object", "probe_2d", "scan_mistakes"):
        shape = opt[key].shape
        d = rng.normal(size=shape)
        if key != "scan_mistakes":
            d = d + 1j * rng.normal(size=shape)
        d = jnp.asarray(d) / np.linalg.norm(d)
        directional_check(loss, opt, key, d)
