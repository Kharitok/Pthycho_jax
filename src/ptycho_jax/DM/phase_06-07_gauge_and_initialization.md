# Phases 6 and 7: Gauge fixing, mode orthogonalization, initialization

---

## Phase 6: Gauge fixing

### Purpose

Several parameter transformations leave the data (and the exit waves) unchanged. Without fixing them, the object scale, the mode norms and the weights drift against each other, and the result is hard to compare or hand off. **Every gauge operation must leave `psi_O = a * S P * E O` unchanged** (the product), and therefore the model intensities.

### Gauge table

| Ambiguity | Fix |
|---|---|
| Mode scale vs weights | Unit-norm modes. For each k: `nu = norm(P_k)`; `P_k <- P_k/nu`; `a_jk <- a_jk * nu` |
| Object scale vs weights | `s = mean(abs(O))` over the covered region; `O <- O/s`; `a <- a * s` |
| Global object phase | `phi = angle(mean(O over covered region))`; `O <- O * exp(-i*phi)`; **compensate** with `P_k <- P_k * exp(+i*phi)` for all k |
| Unitary mixing of modes | Exact symmetry only if weights do not depend on position. With per-position weights this is a canonicalization heuristic (see orthogonalization below) |
| Global translation, and object/probe linear phase ramp | These are true gauges of far-field ptychography: shifting probe and object together translates the exit wave (the far-field modulus is unchanged); a linear phase ramp on the object compensated by the opposite ramp on the probe gives the same intensity up to a per-position constant phase. **Do not fix them inside the algorithm by default.** Remove them in the evaluation metrics (Phase 10). Optional, off by default: probe re-centering if drift is observed |

**Covered region** = canvas pixels with coverage count >= 1, preferably excluding the lowest-coverage edge (for example coverage >= 0.5 times the median count). Define it once from the geometry.

### Mode orthogonalization (optional, `cfg.ortho_every`, 0 = off)

1. **Weighted modes:** `Q_k = sqrt(wbar_k) * P_k`, with `wbar_k = mean_j(a_jk^2)`.
2. **Gram matrix:** `G_kl = <Q_k, Q_l>` over all pixels (K x K, Hermitian).
3. **Eigendecomposition**, sorted by **descending** eigenvalue; eigenvectors `U` (rows or columns consistently).
4. **Rotate** the mode axis of psi: `psi'_jk = sum_l U_kl psi_jl`. This leaves `sum_k abs(F psi_jk)^2` exactly invariant, so P_F is unaffected (check C4.7).
5. **New modes:** `P'_k = (sum_l U_kl Q_l) / sqrt(lambda_k)` with `lambda_k` the eigenvalues; unit norm by construction.
6. **New weights:** keep each position's total flux: `a'_jk = sqrt(lambda_k) * c_j` with `c_j = sqrt( sum_k a_jk^2 / sum_k wbar_k )`.
7. **Refit:** immediately run the weight block (`solve_weights`, per_position mode if active) with the rotated psi and new P' to refresh `a`.

Position-independent weights make this an exact gauge change. With per-position weights, steps 6 and 7 re-establish consistency approximately.

### Mode-collapse guard (`collapse_guard`)

If mode k has `max_j a_jk` below `1e-3` times the largest mode weight, **freeze** it (stop updating it) or **re-seed** it (`P_k <- small random perturbation of the leading mode`, orthogonalized against the others; set `a_jk` to a small starting value). Choose the behaviour via `cfg.collapse_action`. Log every event in the metrics (Phase 9).

### Checks

| ID | Check | Tolerance |
|---|---|---|
| C6.1 | Each gauge operation (mode normalization, object scale, object phase with compensation) leaves `psi_O` unchanged | tol64 |
| C6.2 | Model intensities (Phase 1 stage model) unchanged by each gauge operation | tol64 |
| C6.3 | After normalization: all `norm(P_k) == 1`; `mean(abs(O))` over the covered region is 1; mean phase of O over the covered region is 0 | 1e-10 |
| C6.4 | **Orthogonalization, constant weights:** the Gram matrix of the new weighted modes is diagonal with descending eigenvalues; `sum_k abs(F psi'_jk)^2` equals the same before and after | 1e-10 |
| C6.5 | **Orthogonalization, per-position weights:** after step 7, the model error is not worse than before (relative tolerance 1e-3); report the change | reported / 1e-3 |
| C6.6 | **Collapse guard:** an engineered case with one zero-weight mode triggers the configured action and the event is logged | exact |
| C6.7 | Eigenvalue ordering is stable under a random unitary mixing of the input modes (same sorted eigenvalues) | 1e-10 |

### Gate

C6.1 to C6.7 pass.

---

## Phase 7: Initialization

### Steps

1. **Probe modes** [K, Ny, Nx]
   - Mode 0: a user-supplied or previously estimated probe if available. Otherwise `P0 = ifft2c(sqrt(mean_j(I_meas_j)))` (zero-phase estimate; it is focused and peaky, so allow an optional support or aperture mask, or a mild quadratic phase, via config).
   - Modes 1 to K-1: `P0 * (1 + eps_k * r_k)`, with random complex fields `r_k` (optionally low-pass filtered) and `eps_k` between 0.01 and 0.05. Then **orthonormalize** the stack (QR on the flattened modes, keeping mode 0 as the first direction). **Different seeds per mode; identical modes would stay identical (C5.11).**
   - Normalize every mode to unit norm.
2. **Object:** ones (complex) on the full canvas.
3. **Weights:** a mode profile (default `[1, 0.1, 0.05, ...]`, normalized to sum 1, configurable) times each position's measured counts: `w_jk = profile_k * counts_j`, `a = sqrt(w)`. With unit-norm modes, an ortho FFT and an object of ones, the per-position total modeled power equals the measured counts.
4. **Exit waves:** `psi = exit_waves(P, O, a, geom)`.
5. **Iteration counter** `it = 0`.

### Checks

| ID | Check | Tolerance |
|---|---|---|
| C7.1 | Per-position modeled total intensity equals the measured total counts (on valid pixels, accounting for any masked area) | 1e-6 relative |
| C7.2 | Modes are orthonormal: `Gram == I` | 1e-10 |
| C7.3 | Mode 0 equals the supplied or estimated probe (up to normalization) | tol64 |
| C7.4 | Different seeds give different higher modes; the same seed reproduces exactly | exact |
| C7.5 | The initial model error and Fourier error are recorded as the **baseline** that DM must beat | recorded |

### Gate

C7.1 to C7.5 pass.
