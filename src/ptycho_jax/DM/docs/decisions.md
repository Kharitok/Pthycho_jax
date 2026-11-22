# Implementation decisions

## 2026-11-22 — Match the existing fixed-position object shift exactly

The reference `prepare_correcting_sample_selector` extracts `(Ny+2m,Nx+2m)`, applies a cosine edge taper, Fourier-shifts it by `-fraction`, and takes the central frame. Moving a band-limited object patch and multiplying by a probe is not generally identical to Fourier-shifting the probe on a different finite grid. The user's request explicitly requires the current integer-slicing/Fourier-shift convention. Therefore the DM model is `psi_jk = a_jk P_k L_j O`, where `L_j = crop * shift(-delta_j) * taper * extract_j`. The existing AD model is untouched. Numerical forward/gradient and handoff equality are required; no 1e-3 mismatch allowance is used.

The normal operator for O is not diagonal at fractional positions. Use exact matrix-free, damped preconditioned CG for that block. The probe normal operator is diagonal for all positions in this formulation. The object normal diagonal can be computed with the squared circular-shift kernel and used as a preconditioner; integer positions admit a direct diagonal object solve.

## 2026-11-22 — Resolve gating and normalization contradictions in the guidance

Gauge normalization can change probe/weight arrays even when the corresponding optimization gate is frozen. To preserve the guide's strict frozen-block checks, perform compensated mode/object gauge normalization inside overlap updates only once both probe and weights are enabled. Always perform an exact compensated normalization for final handoff. Initially fixed per-shot flux follows the measured counts as required by phase 7; a later global-weight stage intentionally replaces it by a common profile before the per-shot stage.

Mode mixing is not an exact gauge for per-shot diagonal weights. Orthogonalization remains off in production DM. Optional constant-weight canonicalization must preserve the incoherent intensity; no per-shot heuristic rotation is silently applied.

## 2026-11-22 — Parameter scaling at AD handoff

DM stores unit-norm modes and positive amplitude coefficients `a`. The reference AD model stores unnormalized modes and complex `(shots,K,K)` amplitude matrices. Move `sqrt(mean(a**2))` into each probe mode and put `a/scale` on each shot matrix's diagonal. Fields and intensities are unchanged, while parameter magnitudes match the validated Adam learning rates. AD positions are placed in fixed parameters and injected into the reference model by a wrapper; they are absent from optimizer leaves.

## 2026-11-22 — GPU contraction precision and AD runner dtype

On the installed Blackwell GPU the default reference mode-mixing contraction uses approximate tensor math. A float32 reference-generated case differed from elementwise-equivalent DM intensities by 3.09e-4 relative, creating a spurious noiseless amplitude R=5.28e-8. Requesting highest contraction precision reduced the intensity discrepancy to 1.01e-7 and R to 7.65e-15. New DM contractions and the reference-model wrappers explicitly request highest precision; the shared AD source is untouched. This is a numerical setting, not a relaxed handoff tolerance.

The existing fused AD full-pass runner initializes its loss carry as float32. When x64 is enabled, the reference selector's untyped frequency grids can promote its loss to float64 even for complex64 parameters. Float64-enabled AD tests therefore use the existing streaming full-pass variant; production x64-disabled complex64 runs use the fused full-pass runner. Both accumulate the same per-shot loss and gradient with positions frozen.

## 2026-11-22 — Synthetic evaluation support and fixed-weight controls

A/B fixed-weight cases freeze the known generating amplitudes. Freezing measured transmitted counts instead would prescribe incorrect incident flux because the true object is absorbing; that is a distinct negative-control problem. F_fixed freezes true mean amplitudes against varying truth. Synthetic object errors are evaluated on a fixed truth normal-diagonal support above 5% maximum, while the full geometric-coverage error is retained as an additional metric. Gauge alignment fits only covered pixels using complex residuals; it does not unwrap arbitrary phases in uncovered regions. Translation alignment is not implemented and is stated explicitly. Recovery thresholds remain unchanged and unmet targets are reported.
