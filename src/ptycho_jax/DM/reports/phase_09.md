# Phase 09 implementation report

M1, update norm, current overlap-model MSE/R, mode-weight statistics and inactive modes are recorded each iteration. Raw DM iterates are not mistaken for the fitted overlap model at handoff.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C9_1_C9_2_C9_3_metrics_match_independent_values` | PASS |

| Measured quantity | Relative value | Requirement |
| --- | --- | --- |
| MSE handoff error | 3.1266e-15 | < 1e-12 |

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
