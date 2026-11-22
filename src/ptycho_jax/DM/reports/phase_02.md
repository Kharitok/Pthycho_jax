# Phase 02 implementation report

Crop origins and fractions are fixed arrays. Enlarged patches are validated before compilation; no persistent per-pixel index arrays are stored. On-demand broadcast scatter is used, with an independent scan implementation available.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C0_5_weights_and_C2_5_position_roundtrips` | PASS |
| `test_C1_1_C2_1_C2_2_exact_reference[False]` | PASS |
| `test_C1_1_C2_1_C2_2_exact_reference[True]` | PASS |
| `test_C2_3_invalid_crop_rejected` | PASS |

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
