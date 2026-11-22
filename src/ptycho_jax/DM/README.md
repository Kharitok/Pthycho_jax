# Difference-Map (DM) ptychography from the existing AD forward model: implementation plan

**Audience:** the implementing agent. Read this file completely, then work through the phase documents in order.

## 1. Goal and scope

Build a Difference-Map reconstruction in JAX that reuses the existing automatic-differentiation (AD) forward model. Its output is a good starting guess for the existing optax optimization.

**DM optimizes:**
- the object O
- the probe modes P_k
- the modal weights w_jk, **one set of K weights per scan position j**

**Out of scope (do not implement):**
- Scan-position correction. Positions are fixed inputs everywhere.
- Any change to the existing AD model. It stays as the untouched reference.
- Noise-aware or relaxed modulus constraints. They are a possible later extension.

## 2. Documents (work in this order)

| File | Phases | Content |
|---|---|---|
| `phase_00_conventions_audit.md` | 0 | Audit the AD model, fill the conventions sheet, golden reference data |
| `phase_01-02_forward_stages_and_geometry.md` | 1, 2 | Split the forward model into stages; fixed-position geometry |
| `phase_03_operators_and_adjoints.md` | 3 | Linear operators, Hermitian adjoints, denominators |
| `phase_04_fourier_projection.md` | 4 | Modulus (Fourier) projection P_F |
| `phase_05_overlap_projection.md` | 5 | Overlap projection P_O for object, probe, weights (core piece) |
| `phase_06-07_gauge_and_initialization.md` | 6, 7 | Gauge fixing, mode orthogonalization, initialization |
| `phase_08-09_dm_loop_and_monitoring.md` | 8, 9 | DM step, schedule, loops, metrics |
| `phase_10_validation_ladder.md` | 10 | Synthetic validation cases A to F |
| `phase_11-12_handoff_and_scaling.md` | 11, 12 | Hand-off to optax, memory and multi-device scaling |

## 3. The model DM solves

```
psi_jk = a_jk * (S_j P_k) * (E_j O)          a_jk = sqrt(w_jk) >= 0
I_j(q) = sum_k | F[ psi_jk ](q) |^2
```

| Symbol | Meaning | Shape |
|---|---|---|
| J, K | number of scan positions, number of probe modes | scalars |
| (Ny, Nx) | probe / detector frame size | |
| (Hc, Wc) | object canvas size | |
| O | object (complex) | [Hc, Wc] |
| P | probe modes | [K, Ny, Nx] |
| a | square roots of the weights, **the stored parameter** | [J, K] |
| psi | exit waves, the DM iterate | [J, K, Ny, Nx] |
| I_meas, M | measured intensities, valid-pixel mask | [J, Ny, Nx] |
| n_j | integer part of position j (crop origin) | [J, 2] int |
| delta_j | sub-pixel remainder of position j | [J, 2] float |
| E_j | integer-pixel crop of the canvas at n_j | linear |
| S_j | sub-pixel shift of the probe, a Fourier phase ramp from delta_j | linear, unitary |
| F | 2D FFT (normalization as in the AD model, ortho preferred) | |

**Constraint sets**
- **Overlap set** (P_O projects onto it): all psi that can be written as a_jk * S_j P_k * E_j O for some P, O and a >= 0.
- **Fourier set** (P_F projects onto it): all psi with sum_k abs(F psi_jk)^2 = I_meas on valid pixels.

**Why the sub-pixel shift sits on the probe:** shifting the probe by delta instead of the object by -delta changes the exit wave only by a pure translation. That changes the far-field phase but not the far-field intensity, so the model is intensity-equivalent to an AD model that shifts via Fourier phase ramps. If the AD model instead interpolates the object, this is only approximate. Phase 2 quantifies the difference.

**Weights extension:** standard mixed-state DM has position-independent weights. Per-position weights are an extension. The weight update is therefore a **staged feature**: `fixed`, `global`, `per_position`, each validated separately.

## 4. Ground rules (non-negotiable)

1. **Never edit the reference AD model.** Import it, and wrap it if needed.
2. **`jax.grad` is never used to compute DM updates.** AD is allowed for (a) adjoints via `jax.linear_transpose`, (b) test cross-checks, and (c) evaluating the final loss.
3. **Pure functions, pytree state.** No Python-side mutation inside jitted code.
4. **Large arrays are arguments, not closure constants.** Closing over big arrays in a jitted function embeds them in the compiled program. Pass measured data and geometry as arguments.
5. **Static vs traced.** Shapes, K, `n_inner`, solver variants, chunk sizes are static. Schedule gates (probe on/off, weight mode, ortho on/off) are **traced** values derived from the iteration counter, so crossing a stage boundary never recompiles.
6. **Precision.** Tests that check algebraic identities run in float64/complex128 (`jax.config.update("jax_enable_x64", True)`). Production runs in float32/complex64. Check that no silent promotion occurs.
7. **Gate discipline.** Do not start phase N+1 until every check of phase N passes. A failed check is a bug to fix, not a tolerance to loosen. If a tolerance genuinely must change, record why in `docs/decisions.md`.
8. **Document shapes** in every function docstring.

**Tolerance convention used in all documents:** `tol64` = 1e-12 relative (algebraic identities in float64). `tol32` = 1e-5 relative (float32). Other tolerances are given explicitly.

## 5. JAX toolbox (use these instead of hand-rolled loops)

| Need | Tool |
|---|---|
| Per-position independent work | native batching over the leading axis, or `jax.vmap` |
| Adjoint of a linear operator | `jax.linear_transpose`, wrapped by the Hermitian helper (Phase 3) |
| Iteration loop with recorded metrics | `jax.lax.scan` |
| Fixed-length inner loops (`n_inner`) | `jax.lax.fori_loop` |
| Stage gates on traced booleans | `jax.lax.cond`, `jnp.where`, `jax.lax.switch` for the weight mode |
| Memory-bounded chunking over positions | `jax.lax.map` (`batch_size=` in recent JAX versions) or `lax.scan` over chunks with an accumulator carry |
| Avoid copies of the big iterate | `jax.jit(..., donate_argnums=...)` on psi |
| Several devices | `shard_map` or `jax.sharding` over the position axis |
| State container | `NamedTuple` or a registered dataclass pytree |
| Retrace detection | a Python counter incremented inside the traced function body (it increments only on tracing) |
| Matrix-free linear solves (optional probe solver) | `jax.scipy.sparse.linalg.cg` |

## 6. Working protocol

For each phase:
1. Implement the functions listed in the phase document.
2. Implement **one test per check ID** (for example `test_C3_2_dot_test_S`). Test names carry the check IDs.
3. Run the full test suite, including all earlier phases.
4. Write `reports/phase_XX.md` containing: a short summary; a table (check ID, tolerance, measured value, pass/fail); deviations from this plan and why; open questions.
5. Stop and ask the user if a gate fails after reasonable debugging, or if the AD model contains something DM cannot represent (see Phase 0).

Maintain `docs/decisions.md` (append-only log of conventions chosen and deviations).

## 7. Suggested repository layout

```
dm_ptycho/
  conventions.py    # fft2c/ifft2c, weight parametrization converters, hermitian_adjoint helper
  geometry.py       # Geometry pytree: position split, validity checks
  stages.py         # extract_patches, shift_probe, exit_waves, propagate, detector_intensity
  operators.py      # E, E_H, S, S_H, A_O, A_O_H, A_P, A_P_H, denominators
  projections.py    # fourier_projection, solve_object, solve_probe, solve_weights, overlap_projection
  gauge.py          # normalize_modes, fix_object_gauge, orthogonalize_modes, collapse_guard
  init.py           # initial probe modes, object, weights, exit waves
  dm.py             # DMState, DMConfig, dm_step, run_dm
  metrics.py        # DM metrics and synthetic-data evaluation metrics (gauge-aligned)
  handoff.py        # conversion to the AD/optax parameterization
tests/              # one file per phase; golden/ holds reference data
scripts/run_validation_ladder.py
docs/conventions.md  docs/decisions.md
reports/
```

## 8. State and config objects

- `DMState` (pytree): `psi` [J,K,Ny,Nx], `probe` [K,Ny,Nx], `obj` [Hc,Wc], `a` [J,K], `it` (int32 scalar).
- `Geometry` (pytree): `n` [J,2] int32, `delta` [J,2] float, plus canvas shape and frame shape (static). **Do not store full per-pixel index arrays** (see Phase 2).
- `Data` (pytree): `sqrt_I` [J,Ny,Nx], `mask` [J,Ny,Nx], `counts` [J].
- `DMConfig` (frozen dataclass, hashable, passed as a static argument): K, `n_inner`, beta, alpha values, epsilons, solver variants, chunk sizes, schedule thresholds.

## 9. Hyperparameters

| Name | Default | Notes |
|---|---|---|
| beta | 1 | Implement beta = 1 first; general beta afterwards |
| n_inner | 3 | Range 2 to 5, static |
| alpha_O, alpha_P (relative) | 1e-3 | Absolute alpha = relative value times max of the corresponding denominator map |
| eps (P_F amplitude floor) | 1e-8 times sqrt(mean I_meas) | |
| eps_w (weight step) | 1e-8 times mean squared patch norm | |
| n_probe_start | 10 | Object-only before this |
| n_weight_global_start | 30 | |
| n_weight_pp_start | 50 | Per-position weights start here |
| n_ortho | 0 (off) | Enable once the base cases pass |
| weight_blend lambda | 1 | `a_new = (1-lambda)*a_global + lambda*a_fit`, a knob to curb per-position freedom |
| n_iter | 100 to 300 | |
| eval_every | 5 | Interval for the (costlier) model-error metric |

## 10. Known risks (watch and report, do not hide)

- **Weight degeneracy.** Per-position weights can absorb flux variation, model mismatch and object-scale errors. Validation case F and the blend lambda diagnose this.
- **Mode-mixing gauge is not exact** when weights vary with position. Keep orthogonalization optional.
- **Sub-pixel mismatch** if the AD model interpolates the object rather than shifting the probe (Phase 2).
- **Diagonal approximation** in the probe update for sub-pixel shifts (Phase 3 and 5). An optional CG solver is specified.
- **Global translation / linear-phase-ramp gauge** between probe and object (Phase 6). It is a gauge, not a bug.
- **High error plateau** can indicate wrong scan positions. This plan does not fix positions: report it, do not try to fix it here.
- **Memory.** psi is the dominant array (Phase 12).

## 11. Definition of done

1. All checks of all phases pass and reports exist.
2. Validation ladder cases A to F meet the acceptance criteria in `phase_10_validation_ladder.md`, or deviations are explained.
3. Hand-off check (Phase 11): the AD loss at the converted DM parameters equals the DM model error, and a short optax run descends.
4. A run script reproduces the ladder and writes plots and a markdown results table.

## 12. Resolve early (try the code first, then ask the user)

1. Where are the AD forward model, loss and optax driver? Which JAX version?
2. Typical J, K, frame size, canvas size, dtype, available hardware (drives Phase 12).
3. Does the AD model shift via the probe or the object? Is it Fourier phase-ramp or interpolation?
4. How are the weights parametrized and normalized in the AD model?
5. Are there model terms beyond the equation in section 3 (background, per-position flux, detector gain, etc.)?
