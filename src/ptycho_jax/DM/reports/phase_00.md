# Phase 00 implementation report

The reference audit and complex128 golden archive are complete. General full complex AD matrices cannot map back to the restricted nonnegative diagonal DM weights and are explicitly rejected.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C0_1_parseval` | PASS |
| `test_C0_3_analytic_gaussian` | PASS |
| `test_C0_5_weights_and_C2_5_position_roundtrips` | PASS |
| `test_C0_4_integer_crop_sign` | PASS |
| `test_C0_2_golden_reference_and_reproducibility` | PASS |

| Measured quantity | Relative value | Requirement |
| --- | --- | --- |
| Reference intensity | 2.6436e-16 | < 1e-12 |

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
