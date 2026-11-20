# %% [markdown]
# # 3D Bragg ptychography: forward model vs. exact 3D calculation (ITO 400 cube)
#
# Run cell by cell (`# %%`). Everything you may want to change is in the
# "PARAMETERS" cell. Pipeline of this example:
#
#   1. experiment geometry (9 keV, 75 um pixels at 1.35 m, ITO 400 reflection)
#   2. 2D focused probe  ->  3D probe volume (propagate + rotate by 2*theta_B)
#   3. cubic ITO particle defined in the LAB frame -> beam-aligned grid
#   4. scan positions x rocking angles
#   5. FORWARD MODEL (fast, differentiable, what you use in a reconstruction)
#   6. EXACT 3D reference (explicit 3D Fourier transform of probe*rotating object)
#   7. comparison, gradient / position-correction demo, lab-frame view
#
# Frames: sample/lab (x,y,z): z || Q0 (400 direction), rocking axis = y.
#         beam grid (u,v,w): w || k_f, (u,v) = detector plane. The object lives on
#         the (orthogonal) beam-aligned grid, so voxel pitch in u,v is fixed by the
#         detector:  du = dv = lambda * distance / (N_pixels * pixel_size).

# %%
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # remove to use a GPU
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)  # double precision: needed to see ~1e-12 agreement

from bragg.forward import make_bragg_model
from bragg.frames import beam_to_lab
from bragg.geometry import BraggGeometry, delta_Q_beam_frame, scan_to_grid_positions
from bragg.reference import (
    exact_intensity_3d,
    grid_axes,
    lab_function_on_beam_grid,
    soft_cube_phantom,
)

# %% [markdown]
# ## 1. PARAMETERS (edit here)

# %%
# ---- X-ray beam / crystal -------------------------------------------------
ENERGY_KEV = 9.0
LATTICE_A = 10.12e-10  # ITO (In2O3 bixbyite) cubic lattice constant [m]
HKL_N = 4  # 400 reflection -> d = a / 4

# ---- detector --------------------------------------------------------------
PIXEL = 75e-6  # [m]
DET_DIST = 1.35  # sample-detector distance [m]
N_DET = 64  # detector pixels used (N_DET x N_DET window around the Bragg peak)

# ---- probe -----------------------------------------------------------------
BEAM_FWHM = 500e-9  # intensity FWHM of the focused beam [m]

# ---- particle (cube with faces perpendicular to the lab axes) ---------------
CUBE_SIZE = 300e-9  # edge length [m]
EDGE_SOFTNESS = 40e-9  # width of the (smooth) edges [m]; keep >= ~1 voxel (sharper edges make the sub-pixel Fourier shift less exact)
# Slow phase of the particle (e.g. strain: phase = 2*pi*Q.u(r)); coefficients of
# soft_cube_phantom: (quadratic in x, y*z cross term, linear in z). Set to 0 for a flat phase.
PHASE_COEFFS = (1.5, 0.8, 0.6)

# ---- scan (in the SAMPLE frame, positions on the sample surface) ------------
N_SCAN_X, N_SCAN_Y = 5, 5
STEP_X = 0.6e-6  # along sample x [m]  (beam footprint on sample is elongated by 1/sin(theta_B))
STEP_Y = 0.15e-6  # along sample y [m]

# ---- rocking curve (angle relative to the exact Bragg angle) ----------------
N_ROCK = 11
ROCK_STEP = 0.1e-3  # [rad]; angles = (-N/2 .. N/2) * step

# ---- numerical grid ---------------------------------------------------------
VOXEL_W = None  # voxel size along w [m]; None -> same as du
PATCH_W = 32  # voxels along w in a patch (= depth of the 3D probe volume)
OBJ_UV = 128  # object grid size in u,v (must hold patch + scan range + 2*CROP)
CROP = 4  # pixels cut around a patch after the sub-pixel Fourier shift
MAX_CORR = 5.0  # position-correction range in voxels (tanh-bounded)

# %% [markdown]
# ## 2. Geometry derived from the parameters

# %%
wavelength = 12.398419843320026e-10 / ENERGY_KEV  # [m]
d_hkl = LATTICE_A / HKL_N
theta_B = np.arcsin(wavelength / (2 * d_hkl))
ang_pix = PIXEL / DET_DIST  # angular size of a pixel [rad]
du = dv = wavelength / (N_DET * ang_pix)  # real-space voxel (reciprocal-space pixel = 1/(N du))
dw = du if VOXEL_W is None else VOXEL_W

geom = BraggGeometry(wavelength=wavelength, theta_B=theta_B, du=du, dv=dv, dw=dw)
PATCH = (N_DET, N_DET, PATCH_W)  # patch = part of the object seen by one exposure
OBJ = (OBJ_UV, OBJ_UV, PATCH_W)  # object depth = patch depth (the patch follows the particle in w)
pitches = (du, dv, dw)

thetas_rock = (np.arange(N_ROCK) - (N_ROCK - 1) / 2) * ROCK_STEP
dq_det = 1.0 / (N_DET * du)  # detector pixel in reciprocal space [1/m]
dQw_max = np.max(np.abs([delta_Q_beam_frame(t, theta_B, wavelength)[2] for t in thetas_rock]))

print(f"wavelength          : {wavelength*1e10:.4f} A")
print(f"d_400               : {d_hkl*1e10:.4f} A   ->  Bragg angle {np.rad2deg(theta_B):.3f} deg")
print(f"detector pixel      : {ang_pix*1e6:.2f} urad  =  {dq_det:.3e} 1/m")
print(f"voxel (u,v,w)       : {du*1e9:.1f}, {dv*1e9:.1f}, {dw*1e9:.1f} nm")
print(f"patch / object      : {PATCH[0]*du*1e6:.2f} um (u,v) x {PATCH_W*dw*1e9:.0f} nm (w)")
print(f"rocking range       : +-{np.max(np.abs(thetas_rock))*1e3:.2f} mrad")
print(f"max |dQ_w|          : {dQw_max:.3e} 1/m  (sampling limit 1/(2 dw) = {1/(2*dw):.3e})")
assert dQw_max < 1 / (2 * dw), "rocking range too large for this w sampling -> reduce range or dw"
print(f"cube edge           : {CUBE_SIZE/du:.1f} voxels")

# %% [markdown]
# ## 3. Probe: a focused Gaussian beam defined as a 2D field at the focus
# The model takes only the 2D complex field at the focal plane. It is propagated
# (angular spectrum) to a 3D volume and rotated into the beam frame **once per update**,
# not once per scan point.

# %%
w0 = BEAM_FWHM / np.sqrt(2 * np.log(2))  # amplitude exp(-r^2/w0^2): intensity FWHM = w0*sqrt(2 ln 2)
u_ax, v_ax, _ = grid_axes((N_DET, N_DET, 1), pitches)
probe_2d = jnp.exp(-(u_ax[:, None] ** 2 + v_ax[None, :] ** 2) / w0**2).astype(complex)

# %% [markdown]
# ## 4. Particle: cube with faces along the lab axes (z || Q)
# `fn_lab(x, y, z)` is the complex envelope of the particle in the LAB frame (Bragg carrier
# removed; amplitude = shape, phase = strain along Q). To simulate YOUR particle write a
# different `fn_lab`, or sample your own array on the lab grid and use
# `bragg.frames.lab_to_beam`.

# %%
half = CUBE_SIZE / 2


def fn_lab(x, y, z):
    return soft_cube_phantom(x, y, z, half_size=half, softness=EDGE_SOFTNESS,
                             phase_coeffs=PHASE_COEFFS)


# Object on the beam-aligned grid (exact analytic evaluation, no interpolation)
obj = lab_function_on_beam_grid(fn_lab, OBJ, pitches, theta_B).astype(complex)

# View the same object back on the lab grid, by the Fourier-shear transform
obj_lab = beam_to_lab(obj, theta_B, du, dw, pad=32)

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
c = OBJ[1] // 2
ext_u = np.array([-OBJ[0] // 2, OBJ[0] // 2]) * du * 1e6
ax[0].imshow(np.abs(obj[:, c, :]).T, origin="lower", cmap="gray")
ax[0].set_title("|object|, beam grid (u, w) slice")
ax[1].imshow(np.abs(obj_lab[:, c, :]).T, origin="lower", cmap="gray")
ax[1].set_title("same, lab grid (x, z), z || Q")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
c = OBJ[1] // 2
ext_u = np.array([-OBJ[0] // 2, OBJ[0] // 2]) * du * 1e6
ax[0].imshow(np.angle(obj[:, c, :]).T, origin="lower", cmap="jet")
ax[0].set_title("angle(object), beam grid (u, w) slice")
ax[1].imshow(np.angle(obj_lab[:, c, :]).T, origin="lower", cmap="jet")
ax[1].set_title("same, lab grid (x, z), z || Q")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Scan positions x rocking angles
# One diffraction pattern = (scan position, rocking angle). The scan is defined on the sample
# surface; `scan_to_grid_positions` converts it to patch origins in voxels (the patch window
# stays on the particle; the probe is invariant along its own axis).

# %%
X = (np.arange(N_SCAN_X) - (N_SCAN_X - 1) / 2) * STEP_X
Y = (np.arange(N_SCAN_Y) - (N_SCAN_Y - 1) / 2) * STEP_Y
XX, YY = np.meshgrid(X, Y, indexing="ij")
XX, YY = XX.ravel(), YY.ravel()
n_pos = XX.size

# beam centre on the object in voxels, relative to the object centre
beam_vox = np.asarray(scan_to_grid_positions(XX, YY, theta_B, du, dv, dw))  # (n_pos, 3)
corner0 = np.array([OBJ[0] // 2 - PATCH[0] // 2, OBJ[1] // 2 - PATCH[1] // 2, 0.0])
pos_per_scan = corner0 + beam_vox  # patch origin (voxels, float) in the object grid

# all (position, angle) combinations -> flat list of exposures
pos_all = np.repeat(pos_per_scan, N_ROCK, axis=0)  # (n_pos*N_ROCK, 3)
theta_all = np.tile(thetas_rock, n_pos)  # (n_pos*N_ROCK,)
n_exp = theta_all.size
assert pos_all[:, :2].min() >= CROP and pos_all[:, :2].max() + N_DET + CROP <= OBJ_UV, "object too small"
print(f"{n_pos} positions x {N_ROCK} angles = {n_exp} diffraction patterns")

# %% [markdown]
# ## 6. Forward model
# `opt` = parameters that a reconstruction would optimise; `fixed` = measurement metadata.
# `scan_mistakes` are the differentiable position corrections (zero here).

# %%
model = make_bragg_model(geom, PATCH, crop=CROP, max_correction=MAX_CORR)
opt = {"object": obj, "probe_2d": probe_2d, "scan_mistakes": jnp.zeros((n_exp, 2))}
fixed = {"positions": jnp.asarray(pos_all), "thetas": jnp.asarray(theta_all)}

batch = jax.jit(model["batch_model"])
chunks = np.array_split(np.arange(n_exp), max(1, n_exp // 64))
I_fwd = jnp.concatenate([batch(opt, fixed, jnp.asarray(ix)) for ix in chunks])
I_fwd = np.asarray(I_fwd).reshape(n_pos, N_ROCK, N_DET, N_DET)
print("forward model intensities:", I_fwd.shape, " (position, angle, u, v)")

# %% [markdown]
# ## 7. Exact 3D reference
# For each exposure: (a) evaluate the particle analytically on the patch grid at the true
# (fractional) position - no Fourier shift involved; (b) multiply with the 3D probe volume;
# (c) explicit 3D Fourier transform at the reciprocal-space point set by the rocking angle,
# taking the detector plane. Same probe volume as the model, so differences isolate the
# object handling (sub-pixel shift, rocking phase ramp).

# %%
P_ref = model["probe_constructor"](opt)  # 3D probe volume (Nu, Nv, Nw), built once


@jax.jit
def exact_one(pos, theta):
    # patch centre relative to the object centre, in voxels
    origin = pos - jnp.asarray(corner0)
    patch = lab_function_on_beam_grid(
        fn_lab, PATCH, pitches, theta_B, origin_voxel=(origin[0], origin[1], origin[2])
    ).astype(complex)
    return exact_intensity_3d(P_ref, patch, theta, geom)


I_ex = np.stack([np.asarray(exact_one(jnp.asarray(pos_all[i]), theta_all[i])) for i in range(n_exp)])
I_ex = I_ex.reshape(n_pos, N_ROCK, N_DET, N_DET)

# %% [markdown]
# ## 8. Comparison

# %%
scale = I_ex.max()
rel = np.abs(I_fwd - I_ex).max(axis=(2, 3)) / scale
print(f"max |forward - exact| / max(I) over all {n_exp} patterns: {rel.max():.2e}")
print(f"integrated intensity ratio forward/exact: {I_fwd.sum()/I_ex.sum():.6f}")

ip, ir = n_pos // 2, int(np.argmax(I_ex[n_pos // 2].sum(axis=(1, 2))))  # centre position, rocking peak
show = [I_fwd[ip, ir], I_ex[ip, ir], np.abs(I_fwd[ip, ir] - I_ex[ip, ir])]
titles = ["forward model", "exact 3D", "|difference|"]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, im, t in zip(ax, show, titles):
    a.imshow(np.log10(im + 1e-6 * scale), origin="lower", cmap="viridis",
             vmin=np.log10(1e-6 * scale), vmax=np.log10(scale))
    a.set_title(f"{t} (log10)")
    a.set_xlabel("detector v pixel")
    a.set_ylabel("detector u pixel")
plt.tight_layout()
plt.show()

# Rocking curves (total counts) and error map over the scan
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(thetas_rock * 1e3, I_ex[ip].sum(axis=(1, 2)), "o-", label="exact 3D")
ax[0].plot(thetas_rock * 1e3, I_fwd[ip].sum(axis=(1, 2)), "x--", label="forward model")
ax[0].set_xlabel("rocking angle (mrad)")
ax[0].set_ylabel("integrated intensity")
ax[0].legend()
ax[0].set_title("rocking curve, central position")
im = ax[1].imshow(rel.max(axis=1).reshape(N_SCAN_X, N_SCAN_Y), cmap="magma")
ax[1].set_title("max rel. error over angles, per scan position")
plt.colorbar(im, ax=ax[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Differentiability and position correction
# The loss gradient w.r.t. `scan_mistakes` (and object / probe) comes from plain `jax.grad`.
# Here: pretend the recorded positions are wrong by a sub-voxel error. The gradient of the
# loss at zero correction points toward the true position.

# %%
true_err = jnp.asarray([0.8, -0.5])  # voxels, unknown to the "reconstruction"
true_err =  jnp.asarray(np.random.rand(*fixed["positions"].shape))
idx = jnp.arange(ir, n_exp, N_ROCK)[: min(n_pos, 5)]  # a few positions, same rocking angle
fixed_true = {"positions": fixed["positions"] +true_err, "thetas": fixed["thetas"]}
data = model["batch_model"](opt, fixed_true, idx)  # "measured" data at true positions


def loss(mistakes):
    o = dict(opt, scan_mistakes=mistakes)
    pred = model["batch_model"](o, fixed, idx)  # model uses the recorded (wrong) positions
    return jnp.mean((jnp.sqrt(pred) - jnp.sqrt(data)) ** 2)


m0 = jnp.zeros((n_exp, 2))
g = jax.grad(loss)(m0)[idx]
print("d loss / d scan_mistakes (first positions):\n", np.asarray(g))
#%%
# Plain gradient descent on the correction (in practice an optimiser from optimizers.py does this)
grad_fn = jax.jit(jax.grad(loss))
m = m0
for it in range(600):
    gm = grad_fn(m)
    m = m - 2.0 * gm / (jnp.abs(gm).max() + 1e-30) * 0.02
print("recovered correction (voxels):", np.asarray(jnp.tanh(m[idx]) * MAX_CORR),
      " true error:", np.asarray(true_err[idx]))

# %%
