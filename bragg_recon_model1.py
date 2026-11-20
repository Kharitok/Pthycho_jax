# %% [markdown]
# # Bragg ptychography reconstruction with MODEL 1 (per-exposure forward model)
# Every diffraction pattern (scan position x rocking angle) is an independent measurement
# evaluated by `bragg.forward.make_bragg_model`. Data come from `bragg_simulate.py`.
# Structure follows `pipeline.py` (transmission): config -> data -> parameters -> loss ->
# optimiser -> `run_optimization_loop` -> inspection.
# Position refinement: (u, v) only. One correction per scan position, shared by its angles.

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

jax.config.update("jax_enable_x64", False)  # float32; set True for validation

from bragg.forward import make_bragg_model
from bragg.recon_utils import (
    make_projection, thickness_support,
    align_object, corner_positions, fit_probe_scale, initial_object, initial_probe_2d,
    position_error_stats, show_object_slices, tie_model1_corrections,
)
from bragg.simulate import SimulatedData
from loss_and_reg import gauss_loss, join_loss_and_forward_batched
from optim_loop import run_optimization_loop
from optimizers import prepare_optimizer

# %% configuration
DATA = os.environ.get("BRAGG_DATA", "bragg_sim_data.npz")
reconstruction_config = {
    "max_correction_magnitude": 5.0,  # voxels, tanh-bounded
    "shift_margin": 4,  # crop (pixels cut after the sub-pixel shift)
    "batch_size": 33,  # exposures per gradient evaluation (memory)
    "n_epochs": 150,
    "use_positions": None,  # None = all scan positions, or an int
    "probe_fwhm_factor": 1.25,  # initial probe is deliberately wrong
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

# %% read data (+ move to the model's conventions)
sim = SimulatedData.load(DATA)
cfg, geom = sim.cfg, sim.geom
n_pos_all, n_rock = sim.intensities.shape[:2]
n_pos = rc["use_positions"] or n_pos_all
PATCH = (cfg.n_det, cfg.n_det, cfg.patch_w)
OBJ = sim.obj_shape

# exposures ordered position-major: (n_pos * n_rock, N, N)
measured = sim.intensities[:n_pos].reshape(n_pos * n_rock, cfg.n_det, cfg.n_det)
root_measured = np.sqrt(measured)
pos_nominal = corner_positions(sim)[:n_pos]
fixed = {
    "positions": jnp.asarray(np.repeat(pos_nominal, n_rock, axis=0)),
    "thetas": jnp.asarray(np.tile(sim.thetas, n_pos)),
}
detector_mask = np.ones((cfg.n_det, cfg.n_det), bool)
print(f"{n_pos} positions x {n_rock} angles = {measured.shape[0]} patterns; object {OBJ}")

# %% model
model1 = make_bragg_model(geom, PATCH, crop=rc["shift_margin"], max_correction=rc["max_correction_magnitude"])
model = tie_model1_corrections(model1, n_rock)

# %% parameters
pitches = (geom.du, geom.dv, geom.dw)
params = {
    "object": jnp.asarray(initial_object(OBJ, pitches, rc["object_init_radius"])),
    "probe_2d": jnp.asarray(initial_probe_2d(sim, rc["probe_fwhm_factor"]).astype(np.complex64)),
    "scan_mistakes": jnp.zeros((n_pos, 2), jnp.float32),  # arctanh of the correction / max
}
# match the overall scale of the probe to the measured counts
if rc["known_probe"]:
    params["probe_2d"] = jnp.asarray(sim.probe_2d.astype(np.complex64))
sel = jnp.arange(0, measured.shape[0], max(1, measured.shape[0] // 20))
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

# %% results: object vs truth (global shift / phase / scale removed)
obj_aligned, shift, err = align_object(np.asarray(params["object"]), sim.true_object)
print(f"object relative error in the support: {err:.3f}  (alignment shift {shift} voxels)")
show_object_slices(obj_aligned, f"reconstruction (aligned), err={err:.3f}")
show_object_slices(sim.true_object, "truth")

# %% position corrections vs true errors
true_err, est, rms, rms0 = position_error_stats(sim, params["scan_mistakes"], rc["max_correction_magnitude"])
print("rms true error (u,v) [vox]:", rms0, "\nrms residual after correction:", rms[+1], "(sign -: ", rms[-1], ")")
plt.figure(); plt.scatter(true_err[:n_pos, 0], est[:, 0], label="u"); plt.scatter(true_err[:n_pos, 1], est[:, 1], label="v")
plt.plot([-4, 4], [-4, 4], "k--"); plt.xlabel("true error (voxel)"); plt.ylabel("recovered"); plt.legend(); plt.show()

# %% fit of the data
pred = np.asarray(model["batch_model"](params, fixed, jnp.arange(0, measured.shape[0], n_rock * 7)))
meas = measured[::n_rock * 7]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(np.log10(meas[0] + 1)); ax[0].set_title("measured"); ax[1].imshow(np.log10(pred[0] + 1)); ax[1].set_title("model")
plt.show()
