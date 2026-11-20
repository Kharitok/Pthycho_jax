# %% [markdown]
# # Simulation of 3D Bragg ptychography data (direct 3D FFT)
# Produces `bragg_sim_data.npz` that is read by `bragg_recon_model1.py` and
# `bragg_recon_stack.py`. The simulation does NOT use the reconstruction models: analytic
# Gaussian probe on the beam grid, analytic particle physically rotated by the rocking
# angle at the true (error-affected) position, a full 3D `fftn` and the q_w = 0 plane.
# Everything you may want to change is in the first cell.

# %%
import os
import sys

# os.environ["JAX_PLATFORMS"] = "cpu"  # uncomment to force CPU
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", False)  # float32 is accurate enough for the data (see precision.md)

from bragg.simulate import ExperimentConfig, make_particle, simulate_dataset

OUT = "bragg_sim_data.npz"

cfg = ExperimentConfig(
    energy_kev=9.0,
    pixel=55e-6,  # detector pixel [m]
    det_dist=1.35,  # sample-detector distance [m]
    n_det=256,  # detector window: 256 .. 512 (512 needs a lot of memory: patch 512x512xNx)
    d_hkl=2.915e-10,  # beta-Sn (200) lattice spacing [m]; sets the Bragg angle (13.7 deg)
    patch_w=64,  # voxels along w; must hold the particle along the diffracted beam
    n_rock=11,  # rocking angles per scan position
    rock_step=None,  # None -> matched to the 3D-FFT lattice (needed by the stack model's fft3 mode)
    beam_fwhm=300e-9,
    n_scan_x=7, n_scan_y=5, step_x=0.6e-6, step_y=0.2e-6,  # sample-plane scan (x step maps to a small u shift)
    pos_err_uv_vox=1.5, pos_err_w_vox=1.0,  # std of the true scan errors in voxels
    photons_max=1e6,  # total photons of the brightest pattern; None = no noise
    seed=0,
)

particle = make_particle(size=0.5e-6, strain_rms=0.6, amp_var=0.12, seed=1)

geom = cfg.derived()
print(f"wavelength {geom.wavelength*1e10:.4f} A, Bragg angle {np.rad2deg(geom.theta_B):.2f} deg")
print(f"voxel du=dv={geom.du*1e9:.1f} nm, dw={geom.dw*1e9:.1f} nm -> particle ~{0.5e-6/geom.du:.0f} voxels")

# %% simulate (direct 3D FFT per pattern)
sim = simulate_dataset(cfg, particle)
print("intensities", sim.intensities.shape, " rocking step %.3e rad, nx=%d" % (sim.thetas[1] - sim.thetas[0], sim.nx))
print("max total counts per pattern: %.3g" % sim.intensities.sum(axis=(2, 3)).max())

# %% save
sim.save(OUT)
print("saved", OUT)

# %% look at the data
fig, ax = plt.subplots(2, 4, figsize=(15, 7))
ip = sim.intensities.shape[0] // 2
for a, k in zip(ax[0], np.linspace(0, cfg.n_rock - 1, 4).astype(int)):
    a.imshow(np.log10(sim.intensities[ip, k] + 1), cmap="viridis")
    a.set_title(f"pos {ip}, theta={sim.thetas[k]*1e3:.2f} mrad")
rock = sim.clean.sum(axis=(2, 3))
ax[1, 0].plot(sim.thetas * 1e3, rock[ip], "o-")
ax[1, 0].set_title("rocking curve, central position")
ax[1, 1].scatter(sim.nominal_vox[:, 0], sim.nominal_vox[:, 1], label="nominal")
ax[1, 1].scatter(sim.true_vox[:, 0], sim.true_vox[:, 1], label="true", marker="x")
ax[1, 1].set_title("scan positions (voxels, u v)")
ax[1, 1].legend()
n = sim.true_object.shape
ax[1, 2].imshow(np.abs(sim.true_object[:, n[1] // 2, :]).T, origin="lower")
ax[1, 2].set_title("|true object| (u, w)")
ax[1, 3].imshow(np.angle(sim.true_object[:, n[1] // 2, :]).T * (np.abs(sim.true_object[:, n[1] // 2, :]).T > 0.2),
                origin="lower", cmap="twilight")
ax[1, 3].set_title("phase (u, w)")
plt.tight_layout()
plt.show()
