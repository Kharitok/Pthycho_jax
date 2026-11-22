# Difference-map implementation and NanoMAX results

The implemented solver follows the supplied phase guides with the adaptations
recorded in [decisions](docs/decisions.md) and the audited
[conventions](docs/conventions.md). [Usage and commands](USAGE.md) cover the
standalone DM and DM-to-AD pipelines.

## Implementation status

- Implemented fixed-position integer extraction, cosine taper, fractional
  Fourier shift and central crop exactly matching the current AD model.
- Implemented Hermitian adjoints, scatter coverage and exact normal diagonals.
  Fractional-position object updates use damped preconditioned CG; probe updates
  are exactly diagonal in this formulation.
- Implemented the shared multimode modulus projection, alternating overlap
  projection, fixed/global/per-shot nonnegative amplitude solves, compensated
  scale/phase gauges, inactive-mode freezing, and seeded orthonormal modes.
- Implemented beta=1 and general-beta DM, traced stage gates, scan blocks,
  donation, checkpoints and per-iteration residual/weight monitoring.
- Implemented exact balanced-scale AD handoff with fixed geometry outside the
  optimizer. AD continues with the same complex sample/probe and full modal
  matrix model used previously.
- Executed the reference-generated synthetic A–F ladder and controls, actual
  NanoMAX nominal/+30% defocus runs, conservative and original-rate AD controls,
  three-size benchmarks and a four-virtual-device comparison.
- Added 13 [phase reports](reports/phase_00.md), a golden reference archive,
  numerical [measurements](reports/measurements.json) and the full JUnit output.

Optional heuristic per-shot mode rotation, translation alignment in synthetic
metrics, operator-level profiler time attribution and fully streamed raw
exit-wave storage are not implemented. The current chunking bounds adjoint
intermediates; the full DM iterate remains resident. These are explicit scope
limits, not certified phase-guide checks.

## NanoMAX experiment

All 169 frames were reconstructed with a 512 × 512 detector crop, two beam modes,
200 DM iterations and 400 AD epochs. Scan positions remain fixed in both stages.
The DM schedule is object-only for iterations 0–9, probe updates from 10, global
weights from 30, and per-shot weights from 50. Each overlap projection uses three
inner sweeps and at most five CG iterations per object block.

Data geometry remains 7.180 m detector distance, 75 µm detector pixels,
8996.194341304086 eV and wavelength 1.3781849716602226e-10 m. Sample pixels are
25.7691877514 nm. Only the probe initialization defocus is changed between
1.500 mm and 1.950 mm; this can change both sample-plane amplitude and phase.
Measured data, detector geometry and scan positions are identical.

### Default conservative AD continuation

Sample/probe/weight Adam rates are 0.003 / 0.03 / 0.0003 for both warm and naive
starts. MSE is the mean squared amplitude residual on valid pixels.

| Probe guess | Initial MSE | After 200 DM | After 400 AD from DM | AD-only, 400 epochs |
| --- | --- | --- | --- | --- |
| 1.500 mm | 196.908 | 10.2513 | 3.47539 | 33.1805 |
| 1.950 mm (+30%) | 197.597 | 133.255 | 47.2621 | 164.838 |

### Additional control at the earlier pipeline's original AD rates

Rates are 0.03 / 0.3 / 0.003; these trials restart from the same saved DM and naive
parameters, with 400 AD epochs for each. They avoid interpreting a comparison
that depends only on a particularly conservative AD step size.

| Probe guess | DM starting MSE | AD after DM | AD-only control |
| --- | --- | --- | --- |
| 1.500 mm | 10.2513 | 2.81624 | 4.95080 |
| 1.950 mm (+30%) | 133.255 | 36.5863 | 54.8178 |

The nominal-defocus DM sample shows the star pattern; AD sharpens the result and
lowers the data error. The +30% case improves numerically but remains severely
artifacted and has not recovered a trustworthy specimen after this budget.
DM therefore supplies a useful nominal-defocus initializer and helps the
wrong-defocus trial relative to its control, but this experiment does **not**
establish robustness to a 30% defocus error.

DM→AD includes additional DM compute and is not an equal-total-runtime benchmark.
The model family also expands at handoff from real nonnegative diagonal DM
amplitudes to full complex 2 × 2 shot matrices. Both effects are explicit. The
DM/AD handoff MSE relative differences are 9.08e-8 and 1.56e-7, so the transfer
itself introduces no significant convention penalty. Neither stage optimizes
positions; a fixed latent named `scan_mistakes` only encodes known fractions in
the unchanged reference API.

## Results to inspect/download

The complete result folder is
`results/nanomax_dm_ad_comparison/`.

- [Main reconstruction report](../../../results/nanomax_dm_ad_comparison/results.md)
- [Original-rate AD control](../../../results/nanomax_dm_ad_comparison/original_ad_rates_comparison.md)
- [Correct-defocus convergence](../../../results/nanomax_dm_ad_comparison/correct_defocus/convergence.png)
- [Wrong-defocus convergence](../../../results/nanomax_dm_ad_comparison/defocus_1.30x/convergence.png)
- [Synthetic ladder](../../../results/dm_validation_ladder_final/ladder_report.md)
- [Benchmark](../../../results/dm_benchmark/benchmark.md)

Each case has `initial/`, `dm/`, `ad/` and `ad_only/` parameter archives and
sample/probe/weight figures; the original-rate controls have their own folders.
All raw DM residual curves and 50-iteration parameter checkpoints are retained.
The run folder also includes copies of the synthetic ladder and benchmarks.
Original run-source bytes are hash-verified in `source_snapshot/`; a post-run
caption clarification does not change any numerical algorithm.

## Validation outcome and limits

There are 37 passing DM tests, plus the 39 previously existing loader/pipeline/
diagnostics tests. Independent complex128 measurements include reference
intensity error 2.64e-16, Hermitian dot errors below 2.5e-16, Fourier-projection
constraint error 3.49e-16, object/probe LS gradient ratios 1.42e-12 / 2.12e-15,
and AD handoff error 3.13e-15. The integer DM loop also matches an independent
NumPy implementation. Chunked full steps and gate transitions are tested;
four virtual CPU devices agree within 8.6e-15 relative.

Synthetic A/B meet the indicative R and object-error targets. C's measured R is
small, but coherence recovery misses the proposed 0.95 target. D does not meet
every mode-weight correlation target. The noisy E case is still above the
proposed 1.1× noise-floor criterion after 200 DM iterations. F shows that
per-shot weights improve the data fit and illuminated-region object error
relative to frozen mean weights. These incomplete parameter/noise recoveries
remain visible in the ladder report; tolerances have not been loosened to hide
them. A/B are already nearly exact, and a fixed-rate AD trial can slightly
perturb their approximately 1e-6 MSE.

GPU scatter-add reduction order may cause small floating-point differences
between seeded runs. Probe/object scale, phase and translation gauges and weak
modes limit parameter interpretation. No held-out NanoMAX scan or independent
resolution measurement is used. Raw source AD/pipeline files were not changed.
