# Phase 01 implementation report

Stages use the exact existing object-patch shift, rather than a shifted-probe approximation. Value and gradient tests cover all fitted blocks. Production arrays remain float32/complex64.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C1_1_C2_1_C2_2_exact_reference[False]` | PASS |
| `test_C1_1_C2_1_C2_2_exact_reference[True]` | PASS |
| `test_C1_2_gradients_match_reference` | PASS |
| `test_C1_3_jit_vmap_C1_4_dtype` | PASS |

| Measured quantity | Relative value | Requirement |
| --- | --- | --- |
| Reference intensity | 2.6436e-16 | < 1e-12 |

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
