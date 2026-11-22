import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest
from ptycho_jax.DM.geometry import make_geometry

@pytest.fixture
def small_case():
    rng = np.random.default_rng(23)
    geom = make_geometry(np.array([[0,0],[0,3],[3,0],[3,3]]),
                         np.array([[.2,.7],[.6,.1],[.4,.8],[.9,.3]]),
                         (20,24), (8,10), 2)
    probe = jnp.asarray(rng.normal(size=(2,8,10)) + 1j*rng.normal(size=(2,8,10)))
    obj = jnp.asarray(1 + .2*rng.normal(size=(20,24)) + .2j*rng.normal(size=(20,24)))
    a = jnp.asarray(rng.uniform(.6,1.4,(4,2)))
    return geom, probe, obj, a
