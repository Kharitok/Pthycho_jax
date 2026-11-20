import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from bragg.geometry import BraggGeometry

import numpy as np


@pytest.fixture(scope="session")
def geom():
    d = 30e-9
    return BraggGeometry(wavelength=1e-10, theta_B=np.deg2rad(20.0), du=d, dv=d, dw=d)


# probe/patch volume (Pu, Pv, Pw); object volume (Ou, Ov, Ow)
PATCH = (64, 64, 32)
OBJ = (128, 128, 32)
W0 = 150e-9
CUBE = dict(half_size=10 * 30e-9, softness=2 * 30e-9)
