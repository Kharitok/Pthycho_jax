# Simulation and reconstruction pipelines (Bragg ptychography)

## Files
| File | Purpose |
|---|---|
| `bragg/simulate.py` | `ExperimentConfig`, analytic Sn-like particle (strain phase, amplitude variation), direct-3D-FFT simulator, `SimulatedData` (npz save/load) |
| `bragg/recon_utils.py` | initial probe/object, probe-scale fit, object alignment, position-error statistics, tying of model-1 corrections |
| `bragg_simulate.py` | cell script: config -> simulate -> `bragg_sim_data.npz` -> plots |
| `bragg_recon_model1.py` | reconstruction with model 1 (per exposure; (u,v) refinement, one correction per scan position shared by its angles) |
| `bragg_recon_stack.py` | reconstruction with model 2 (rocking stack; (u,v,w) refinement) |
| `tests/test_simulate.py` | simulator vs model, save/load, gradient step |

Run order: `bragg_simulate.py`, then either reconstruction script (cells, `# %%`). Both mirror `pipeline.py`
(`join_loss_and_forward_batched(gauss_loss, ...)`, `prepare_optimizer`, `run_optimization_loop`, "accumulate_full_pass").

## Simulation (independent of the models)
9 keV, 55 um pixel, 1.35 m, n_det = 256 (512 possible but the patch 512x512xNx is memory-heavy), beta-Sn (200)
(theta_B = 13.7 deg), 0.5 um particle. Analytic Gaussian probe on the beam grid; particle physically rotated by the
rocking angle at the true (error-affected) position; full 3D `fftn`, plane q_w = 0; Poisson noise.
Full default size: 35 positions x 11 angles, 20 s on one GPU.

## Validation
- Models fed the true object, probe and true position errors reproduce the simulation to <= 1e-2 (relative L2;
  typically 1e-3). With the opposite sign of the position errors the error is 10-100x larger; this fixes the
  convention: **recovered correction ~ +(true - nominal)** (see `tests/test_simulate.py`).
- A small scale factor between probe and data (photon counts) is fitted once (`fit_probe_scale`).

## Cost (256 detector, 35 positions x 11 angles, one GPU, float32)
Model 2: ~0.8 s/epoch. Model 1: ~3.3 s/epoch.

## Honest status of the reconstruction quality
Both scripts run end-to-end and the data loss decreases strongly (stack: 0.06 -> 0.009 in 300 epochs), but with
the default scan (35 positions, 11 angles, 0.6x0.2 um steps, wrong initial probe, position refinement on) the
object is NOT recovered to the truth (relative error in the support ~0.6) and the position errors are not recovered
either. Experiments on a small dataset showed that even with the true probe and true positions the object error
stays ~0.5, so the problem is under-determination/tuning rather than a model bug: few rocking angles, small
overlap along x (x step 0.6 um shifts u only by 0.28x), and probe/object/position degeneracy. Suggestions:
more overlap and positions, more angles, start with probe and positions frozen (set their lr tiny), enable
them later, add an object support/regularisation (`projection_fn`), tune learning rates.

## Known probe and thickness (max extent along the beam) constraint

Implemented in `bragg/recon_utils.py` (`thickness_support`, `make_projection`) and switched on from the
config of both reconstruction scripts: `known_probe`, `thickness` [m], `lateral_halfwidth` (optional),
`freeze_positions`. The constraint is a slab |w| <= thickness/2 on the beam axis w, applied to the object after
every update through `projection_fn` of `run_optimization_loop`. Used thickness 650 nm (true particle extent
along w ~ 595 nm at the 10 % level, i.e. a ~10 % loose bound). "Known probe" = true simulated probe, fixed
(scale from `fit_probe_scale`, lr 0). Setup: default dataset (35 positions x 11 angles, 256 detector, noisy
1e6 photons), Adam lr 3e-2, float32, GPU. Object error = relative error inside the support after removing
global shift/phase/scale. Position rms: true error (u,v,w) = (1.64, 1.26, 0.84) voxel; "residual" = rms of the
remaining error after correction (lower than true error = improved).

| Run | Model | Probe | Thickness | Positions | Epochs | Final loss | Object err | Residual pos. rms (u,v,w) |
|---|---|---|---|---|---|---|---|---|
| ref | stack | unknown | no | refined | 300 | 0.0094 | 0.62 | 1.90, 1.18, 1.37 |
| B | stack | unknown | yes | refined | 300 | 0.0100 | 0.60 | 1.90, 1.30, 1.20 |
| A | stack | known | no | refined | 300 | 0.0117 | 0.28 | 1.00, 0.92, 1.50 |
| C | stack | known | yes | refined | 300 | 0.0117 | 0.22 | 1.06, 0.68, 1.17 |
| D | stack | known | yes | frozen (0) | 300 | 0.0202 | 0.22 | 1.64, 1.26, 0.84 (uncorrected) |
| E | stack | known | yes | refined | 900 | 0.0110 | **0.196** | **0.56, 0.37**, 1.03 |
| F | model 1 | known | yes | refined (u,v) | 120 | 0.0194 | 0.32 | 2.5, 2.6 (worse than none) |

Conclusions
- The probe is the main blocker: knowing it halves the object error (0.6 -> 0.28). The thickness bound adds a
  further modest gain (0.28 -> 0.22) and only helps once the probe is known.
- With known probe, position refinement is needed and works in (u,v): the loss drops from 0.020 (frozen) to
  0.011, and the u,v residual falls from (1.64, 1.26) to (0.56, 0.37) voxel after 900 epochs. The w error is
  not recovered (0.84 -> ~1.0): w shifts are weakly constrained by the data (the stack shifts along w only
  through the rocking-curve phase), so consider freezing or bounding w (`max_correction_magnitude[2]`).
- The object error is still ~0.2 and not converged at 900 epochs (0.224 at 300, 0.196 at 900); the noise
  level (1e6 photons) and the limited rocking range also bound it. Longer runs, lr decay, or a tighter
  support (`lateral_halfwidth`) are the next levers.
- Model 1 is ~2.5x slower per epoch (1.96 vs 0.73 s) and, in 120 epochs, converges more slowly and its
  position corrections are not yet reliable (only two refined parameters per position tied across angles, but
  fewer effective epochs). Run it longer before judging; model 2 is the recommended path.

## Noise-free / perfect positions / more rocking angles

Setup: model 2 (stack), known probe (fixed), thickness bound 650 nm, positions perfect (simulated with zero
position error and not refined), 35 positions, 256 detector, Adam lr 3e-2. Object error as defined above.
Data generated by `ExperimentConfig(pos_err_uv_vox=0, pos_err_w_vox=0, photons_max=None|1e6, n_rock=11|41)`;
the reconstruction script reads the file named in env `BRAGG_DATA`.

| Rocking angles (range) | Noise | Epochs | Final loss | Object error |
|---|---|---|---|---|
| 11 (1.6 mrad) | 1e6 photons | 300 | 1.1e-2 | 0.215 |
| 11 | none | 300 | 6.5e-5 | 0.211 |
| 11 | none | 900 | 4.4e-6 | 0.178 |
| 41 (6.3 mrad) | 1e6 photons | 300 | 5.7e-3 | 0.105 |
| 41 | none | 300 | 2.4e-5 | 0.101 |
| 41 | none | 900 | 3.8e-7 | 0.087 |
| (ref.) 11, noisy, true position errors refined, 900 ep. | 1e6 | 900 | 1.1e-2 | 0.196 |

Conclusions
- Noise is not the limiting factor: removing it (loss drops by 2-3 orders) leaves the object error unchanged at
  300 epochs (0.215 -> 0.211). Perfect positions with refinement-free fitting give about the same error as
  refined noisy positions (0.22 / 0.196), so the position step costs little once the probe is known.
- Clean data converge slowly toward the truth (0.211 -> 0.178 in 900 epochs; the data loss still falls by
  10x), i.e. the problem is poorly conditioned along the weakly constrained directions (limited angular
  coverage), not stuck.
- Angular coverage is what matters: 41 angles (4x wider rocking range, 0.83 vs 0.72 s/epoch with the stack
  model) halves the object error (0.21 -> 0.10 at 300 epochs, 0.087 at 900). The wider range samples more of the
  3D reciprocal space around the Bragg peak (finer resolution along w, better-constrained thickness direction).
- Caveat: the envelope-rotation approximation error grows as theta^2; at +-3 mrad it is still small and the
  model fits the clean data to loss 4e-7, so it is not a limiting factor here.

## 71 rocking angles (range 11 mrad)

Same setup as above (stack model, known fixed probe, 650 nm thickness bound, perfect positions, 300 epochs,
batch 3 stacks, 1.13 s/epoch). 71 instead of 70 angles: with an even number of angles, symmetric about the
Bragg angle, the angles fall half a q_x pixel off the 3D-FFT lattice and `fft3` refuses them (use an odd count
or `stack_mode="dft"`).

| Angles | Noise | Final loss | Object error |
|---|---|---|---|
| 11 | 1e6 | 1.1e-2 | 0.215 |
| 41 | 1e6 | 5.7e-3 | 0.105 |
| 71 | none | 5.4e-6 | 0.095 |
| 71 | 1e6 | 3.4e-3 | 0.087 |

Going from 41 to 71 angles gives only a small further gain (0.105 -> 0.087-0.095, 300 epochs); the error
seems to saturate near ~0.09. The residual is unlikely to be noise (clean and noisy agree). Candidates are the
theta^2 envelope-rotation approximation, which now reaches +-5.5 mrad, the Ewald-curvature neglect, and
the soft particle edges (the error is measured inside a 10 % support). The 41-angle range already captures most of the benefit.

## Model 1 (per-pattern 2D, phase ramp) vs model 2 (3D stack, fft3) at 71 angles

Identical setup: 71 angles, known fixed probe, 650 nm thickness bound, perfect positions (not refined),
300 epochs, Adam 3e-2, same simulated data. Model 1 needs batch 35 (batch 71 ran out of 11 GB GPU memory).

| Model | Noise | Final loss | Object error | Time/epoch |
|---|---|---|---|---|
| 2 (stack, fft3) | none | 5.4e-6 | 0.095 | 1.13 s |
| 1 (2D + phase ramp) | none | 1.0e-5 | 0.089 | 11.7 s |
| 2 (stack, fft3) | 1e6 | 3.4e-3 | 0.087 | 1.13 s |
| 1 (2D + phase ramp) | 1e6 | 3.4e-3 | 0.081 | 11.7 s |

Model 1 is slightly better in object error (0.089/0.081 vs 0.095/0.087; ~6-8 % relative), at ~10x the cost. The
data loss is the same (noisy) or ~2x higher (clean). The difference is small and comes from single runs, so it is
at most a marginal effect. Both models plateau near 0.08-0.09, so the plateau is not caused by the
3D-FFT lattice of model 2; it comes from something shared (envelope-rotation / Ewald approximations, edge
definition of the error metric, or conditioning). Model 2 remains the practical choice.
