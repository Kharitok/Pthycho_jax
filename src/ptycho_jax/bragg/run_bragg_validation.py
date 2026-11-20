"""Validation run: fast Bragg forward model vs exact 3D Fourier transform for a small
cube with a slow phase. Writes bragg_validation.png and prints an error table.

    JAX_PLATFORMS=cpu .venv/bin/python run_bragg_validation.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from bragg.forward import make_bragg_model
from bragg.geometry import BraggGeometry
from bragg.reference import exact_intensity_3d, grid_axes, rotated_phantom, sample_phantom

d = 30e-9
geom = BraggGeometry(1e-10, np.deg2rad(20.0), d, d, d)
PATCH, OBJ, W0 = (64, 64, 32), (128, 128, 32), 150e-9
CUBE = dict(half_size=10 * d, softness=2 * d)
pitches = (d, d, d)

u, v, _ = grid_axes(PATCH[:2] + (1,), pitches)
f0 = jnp.exp(-(u[:, None] ** 2 + v[None, :] ** 2) / W0**2).astype(complex)
model = make_bragg_model(geom, PATCH, crop=4)
obj = sample_phantom(OBJ, pitches, **CUBE).astype(complex)
opt = {"object": obj, "probe_2d": f0, "scan_mistakes": jnp.zeros((1, 2))}
P = model["probe_constructor"](opt)

thetas = np.array([0.0, 2.5e-4, 5e-4, 1e-3, 1.5e-3])
pos = jnp.array([[32.0, 32.0, 0.0]])
rows = []
fig, ax = plt.subplots(3, len(thetas), figsize=(3 * len(thetas), 8))
for j, th in enumerate(thetas):
    fx = {"positions": pos, "thetas": jnp.array([th])}
    I = model["point_model"](opt, fx, 0)
    static = exact_intensity_3d(P, obj[32:96, 32:96, :], th, geom)
    rot = exact_intensity_3d(
        P, rotated_phantom(PATCH, pitches, th, geom.theta_B, **CUBE), th, geom
    )
    e_static = float(jnp.max(jnp.abs(I - static)) / jnp.max(static))
    e_rot = float(jnp.max(jnp.abs(I - rot)) / jnp.max(rot))
    rows.append((th, e_static, e_rot))
    ax[0, j].imshow(np.log10(np.asarray(I.T) + 1e-12), origin="lower")
    ax[0, j].set_title(f"model  theta={th * 1e3:.2f} mrad")
    ax[1, j].imshow(np.log10(np.asarray(static.T) + 1e-12), origin="lower")
    ax[1, j].set_title("exact 3D (static envelope)")
    ax[2, j].imshow(np.log10(np.asarray(rot.T) + 1e-12), origin="lower")
    ax[2, j].set_title(f"exact 3D (rotated), err={e_rot:.1e}")
plt.tight_layout()
plt.savefig("bragg_validation.png", dpi=80)

print("theta[mrad]  rel.err vs exact 3D (static)   rel.err vs exact 3D (rotated envelope)")
for th, a, b in rows:
    print(f"{th * 1e3:10.2f}  {a:24.2e}  {b:32.2e}")
