"""The Bragg point model plugs into the existing loss/optimizer stack unchanged."""

import jax
import optax
import jax.numpy as jnp
import numpy as np

from loss_and_reg import gauss_loss, join_loss_and_forward_batched
from optimizers import prepare_optimizer

from test_batch_and_grad import N, make


def test_existing_batched_loss_and_optimizer_step(geom):
    model, opt, fx = make(geom)
    idx = jnp.arange(N)
    measured = model["batch_model"](dict(opt, scan_mistakes=jnp.zeros((N, 2))), fx, idx)
    root = jnp.sqrt(measured)
    mask = jnp.ones(measured.shape[1:], bool)

    loss = join_loss_and_forward_batched(gauss_loss, model["point_model"])
    loss_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0))
    val, grads = loss_and_grad(opt, fx, idx, root, mask)
    assert jnp.isfinite(val)
    assert set(grads) == set(opt)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads.values())

    optimizer = prepare_optimizer(
        {k: {"type": "adam", "learning_rate": 1e-5} for k in opt}
    )
    state = optimizer.init(opt)
    updates, state = optimizer.update(grads, state, opt)
    new = optax.apply_updates(opt, updates)
    assert loss_and_grad(new, fx, idx, root, mask)[0] < val


def test_probe_not_batched_under_vmap(geom):
    """Existing vmap-over-point_model pattern must not replicate the 3D probe per position."""
    model, opt, fx = make(geom)
    jaxpr = jax.make_jaxpr(
        lambda o: jax.vmap(lambda i: model["point_model"](o, fx, i))(jnp.arange(N))
    )(opt)
    import re

    shapes = re.findall(r"\w+\[([\d,]+)\]", str(jaxpr))
    dims = [tuple(int(x) for x in sh.split(",")) for sh in shapes]
    # probe native stack has Nw + 2*extra_w = 116 planes; no intermediate may carry N x 116
    assert any(116 in d for d in dims)
    assert not [d for d in dims if d[0] == N and 116 in d]
