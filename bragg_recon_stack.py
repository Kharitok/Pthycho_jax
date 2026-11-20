# %% [markdown]
# # Bragg ptychography reconstruction with MODEL 2 (rocking-stack forward model)
# One forward call per scan position produces the whole rocking stack (one 3D FFT or exact
# DFT, `bragg.stack.make_bragg_stack_model`). Position refinement in XYZ (u, v, w voxels) per stack.
# Data come from `bragg_simulate.py`; structure follows `pipeline.py`.

# %%
import os
import sys
import time

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
# os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", False)

from bragg.recon_utils import (
    make_projection, thickness_support,
    align_object, corner_positions, fit_probe_scale, initial_object, initial_probe_2d,
    position_error_stats, show_object_slices,
)
from bragg.simulate import SimulatedData
from bragg.stack import make_bragg_stack_model
from loss_and_reg import gauss_loss, join_loss_and_forward_batched
from optim_loop import run_optimization_loop
from optimizers import prepare_optimizer

# %% configuration
DATA = os.environ.get("BRAGG_DATA", "bragg_sim_data.npz")
reconstruction_config = {
    "stack_mode": "fft3",  # "fft3" (one 3D FFT, needs matched dw/rocking step) or "dft" (any angles)
    "max_correction_magnitude": (5.0, 5.0, 3.0),  # voxels (u, v, w), tanh-bounded
    "shift_margin": 4,
    "batch_size": 5,  # stacks per gradient evaluation (each = n_rock patterns)
    "n_epochs": 150,
    "use_positions": None,
    "probe_fwhm_factor": 1.25,
    "object_init_radius": 0.3e-6,
    "lr": {"object": 3e-2, "probe_2d": 3e-2, "scan_mistakes": 3e-2},
    "use_multigpu": False,
    "known_probe": False,  # True: start from the true probe and keep it fixed
    "thickness": None,  # known sample thickness along the beam w [m] -> slab support; None = off
    "lateral_halfwidth": None,  # optional known (u,v) half-width [m] of the support
    "freeze_positions": False,  # keep scan corrections at zero (only for diagnostics)
}
rc = reconstruction_config
import json
rc.update(json.loads(os.environ.get("BRAGG_RECON_OVERRIDE", "{}")))  # e.g. for experiment drivers

# %% data
sim = SimulatedData.load(DATA)
cfg, geom = sim.cfg, sim.geom
n_pos = rc["use_positions"] or sim.intensities.shape[0]
PATCH = (cfg.n_det, cfg.n_det, cfg.patch_w)
OBJ = sim.obj_shape
measured = sim.intensities[:n_pos]  # (n_pos, n_rock, N, N): one 3D array per scan position
root_measured = np.sqrt(measured)
fixed = {"positions": jnp.asarray(corner_positions(sim)[:n_pos])}
detector_mask = np.ones((cfg.n_det, cfg.n_det), bool)  # broadcast over the rocking axis
print(f"{n_pos} stacks of {measured.shape[1]} angles; object {OBJ}")

# %% model
model = make_bragg_stack_model(
    geom, PATCH, sim.thetas, nx=sim.nx, mode=rc["stack_mode"],
    crop=rc["shift_margin"], max_correction=rc["max_correction_magnitude"],
)

# %% parameters
pitches = (geom.du, geom.dv, geom.dw)
params = {
    "object": jnp.asarray(initial_object(OBJ, pitches, rc["object_init_radius"])),
    "probe_2d": jnp.asarray(initial_probe_2d(sim, rc["probe_fwhm_factor"]).astype(np.complex64)),
    "scan_mistakes": jnp.zeros((n_pos, 3), jnp.float32),
}
if rc["known_probe"]:
    params["probe_2d"] = jnp.asarray(sim.probe_2d.astype(np.complex64))
sel = jnp.arange(0, n_pos, max(1, n_pos // 8))
params, ratio = fit_probe_scale(model["batch_model"], params, fixed, sel, measured[np.asarray(sel)])
print(f"probe scaled by sqrt({ratio:.3g})")

# the fitted scale is applied to the probe; keep that scaled probe fixed when it is "known"
support = None
if rc["thickness"] is not None:
    support = thickness_support(OBJ, pitches, rc["thickness"], rc["lateral_halfwidth"])
    params["object"] = params["object"] * jnp.asarray(support)
frozen = {}
if rc["known_probe"]:
    frozen["probe_2d"] = params["probe_2d"]
    rc["lr"] = dict(rc["lr"], probe_2d=0.0)
if rc["freeze_positions"]:
    frozen["scan_mistakes"] = params["scan_mistakes"]
    rc["lr"] = dict(rc["lr"], scan_mistakes=0.0)
projection = make_projection(support, frozen)

# %% loss, optimiser
get_loss = jax.jit(join_loss_and_forward_batched(gauss_loss, model["point_model"]))
get_loss_and_grad = jax.jit(jax.value_and_grad(get_loss, argnums=0))
optimizer = prepare_optimizer({k: {"type": "adam", "learning_rate": v} for k, v in rc["lr"].items()})
opt_state = optimizer.init(params)

# %% before
show_object_slices(params["object"], "initial object")

# %% reconstruction loop
t0 = time.time()
params, fixed, opt_state, loss_hist = run_optimization_loop(
    params=params,
    non_diff_params=fixed,
    opt_state=opt_state,
    optimizer=optimizer,
    loss_and_grad_fn=get_loss_and_grad,
    measured_batch_pool=root_measured,
    mask=jnp.asarray(detector_mask),
    mode="accumulate_full_pass",
    n_steps=rc["n_epochs"],
    batch_size=rc["batch_size"],
    seed=0,
    shuffle_each_epoch=True,
    projection_fn=projection,
    use_multigpu=rc["use_multigpu"],
)
print(f"{time.time()-t0:.0f} s, {(time.time()-t0)/rc['n_epochs']:.2f} s/epoch, final loss {float(loss_hist[-1]):.4g}")
plt.figure(); plt.semilogy(loss_hist); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()

# %% results
obj_aligned, shift, err = align_object(np.asarray(params["object"]), sim.true_object)
print(f"object relative error in the support: {err:.3f}  (alignment shift {shift} voxels)")
show_object_slices(obj_aligned, f"reconstruction (aligned), err={err:.3f}")
show_object_slices(sim.true_object, "truth")

# %% position corrections (u, v, w) vs true errors
true_err, est, rms, rms0 = position_error_stats(sim, params["scan_mistakes"], rc["max_correction_magnitude"])
print("rms true error (u,v,w) [vox]:", rms0, "\nrms residual after correction:", rms[+1])
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for j, n in enumerate("uvw"):
    ax[j].scatter(true_err[:n_pos, j], est[:, j]); ax[j].plot([-4, 4], [-4, 4], "k--"); ax[j].set_title(n)
plt.show()

# %% fit of the data
pred = np.asarray(model["batch_model"](params, fixed, jnp.arange(0, n_pos, 7)))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
k = measured.shape[1] // 2
ax[0].imshow(np.log10(measured[0, k] + 1)); ax[0].set_title("measured"); ax[1].imshow(np.log10(pred[0, k] + 1)); ax[1].set_title("model")
plt.show()
