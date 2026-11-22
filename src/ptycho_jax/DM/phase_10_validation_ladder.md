# Phase 10: Validation ladder on synthetic data

## Purpose

Show that the DM implementation recovers known truth, in increasing order of difficulty, and that the per-position weight extension actually helps. Data are always generated with the **reference AD forward model** (not the DM stage functions), so conventions are cross-checked.

## Synthetic truth

- **Object:** complex, with real structure (amplitude in about [0.6, 1], smooth phase plus fine features; use a test image, not white noise). Canvas large enough for the scan plus the frame.
- **Probe modes:** K_true orthonormal modes with a localized envelope (for example Hermite-Gauss-like or random modes orthogonalized under a Gaussian envelope). Probe localized away from the frame edge.
- **Weights:** `w_jk = base_k * (1 + drift_k(j) + jitter_jk)`, with a smooth drift across the scan and a small random jitter; clipped to be non-negative.
- **Scan:** Fermat spiral or a grid with random offsets; overlap >= 60 to 70%; J about 100 to 400; frames 64x64 for the ladder (small and fast). Integer or sub-pixel positions according to the case.
- **Noise:** none, or Poisson at a realistic dose (case E).
- All seeds are fixed and logged.

## Cases

| Case | Setup | `weight_mode` | Validates |
|---|---|---|---|
| A | K = 1, integer positions, noise-free | fixed | P_F, P_O object and probe blocks, DM loop, gauge |
| B | A with sub-pixel positions | fixed | Shift handling, diagonal probe approximation (and the CG variant if present) |
| C | K = 3, constant weights, noise-free | global | Mode recovery, orthogonalization, gauge |
| D | K = 3, per-position weights (drift plus jitter), noise-free | per_position | Weight extension |
| E | D with Poisson noise at realistic dose | per_position | Robustness |
| F | D run twice: (i) weights frozen at wrong constants (`fixed`); (ii) `per_position` | fixed vs per_position | Whether per-position weights help; weight degeneracy |

## Evaluation metrics (always after gauge alignment)

**Alignment procedure** (applied to the recovered result before measuring):
1. Fit and remove a complex scalar `c` for the object (`c = <O_rec, O_true> / <O_rec, O_rec>` over the covered region).
2. Fit and remove a linear phase ramp of the object relative to truth over the covered region (fit a plane to the phase of `O_rec * conj(O_true)`); apply the opposite ramp to all probe modes.
3. Estimate any global translation by sub-pixel cross-correlation of the leading probe mode amplitude with the true one; apply it to all modes. Report the shift found.

| Metric | Definition |
|---|---|
| Object error | `norm(c * O_rec - O_true) / norm(O_true)` over the covered region, after alignment |
| Model error R | `sum_valid (sqrt(I_model) - sqrt_I)^2 / sum_valid I_meas` with the recovered parameters |
| Noise floor | The same R evaluated with the **true** parameters on the (noisy) data |
| Probe coherence-operator overlap | With `rho = sum_k wbar_k * outer(P_k, conj(P_k))`: `tr(rho_rec rho_true) / sqrt(tr(rho_rec^2) tr(rho_true^2))`, computed from Gram matrices: `tr(rho_1 rho_2) = sum_{k,l} wbar1_k wbar2_l * abs(<P1_k, P2_l>)^2`. Invariant to unitary mixing of modes. 1 means identical |
| Weight correlation | Match recovered to true modes by the largest absolute overlap `abs(<P_rec_k, P_true_l>)` (greedy assignment). If a mode's best overlap is below 0.8, report it as "unmatched" and skip its correlation. For matched modes: Pearson correlation over positions of the **relative** weights `w_jk / sum_k w_jk`, plus the relative error of the mean mode powers |
| Handoff value | AD loss at the converted DM result versus at the naive initialization, and after 50 optax steps from each |

## Acceptance criteria (indicative: tune after seeing the first results and record changes in `docs/decisions.md`)

| Case | Criterion |
|---|---|
| A | Model error R < 1e-3 within 200 iterations; object error < 1e-2 |
| B | R < 3e-3 within 200 iterations (or document the sub-pixel penalty and the effect of the CG probe solver) |
| C | R < 3e-3; probe coherence overlap > 0.95; mean mode power error < 10% |
| D | R < 5e-3; weight correlation > 0.8 for matched modes |
| E | R within 1.1 times the noise floor; the recovered object and probe stay meaningful |
| F | `per_position` gives a clearly lower R than frozen wrong constants **and** an object error that is not worse. If R improves but the weights look like noise or the object worsens, report weight degeneracy and test `weight_blend < 1` |

## Tasks

1. Write the synthetic-data generator (`tests/synthetic.py`).
2. Write `scripts/run_validation_ladder.py`, which runs cases A to F with fixed seeds and writes:
   - `results/ladder_report.md` (a table of all metrics and pass/fail per case)
   - plots: M1/M2/M3 curves, recovered vs true object (amplitude and phase), recovered vs true probe modes, true vs recovered weights versus position, and one scatter plot per matched mode
3. Run each case with beta = 1 first; repeat with another beta only if time permits.
4. Include a **negative control**: case D run with `weight_mode = fixed` at the true mean weights, to separate weight effects from other effects.

## Checks

| ID | Check |
|---|---|
| C10.1 | Generator self-test: reference model on the truth reproduces the stored data; truth lies on both constraint sets (P_F and P_O fixed points) |
| C10.2 | Alignment procedure is idempotent and recovers a known imposed scalar, ramp and shift to within numerical precision |
| C10.3 | Probe coherence overlap is exactly 1 for a unitary-mixed copy of the true modes |
| C10.4 | Cases A to F meet the acceptance criteria, or the deviation is explained in the report |
| C10.5 | Every run is reproducible from its seed |

## Gate

C10.1 to C10.3 pass and the ladder report exists, with each case's outcome stated honestly (including failures).
