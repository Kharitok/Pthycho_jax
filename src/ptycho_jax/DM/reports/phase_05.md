# Phase 05 implementation report

Object solves are proximal CG (direct diagonal for integers); probe and nonnegative amplitude solves are analytical. Frozen blocks remain bit-identical. The guide’s assertion that 20 alternating sweeps must recover all bilinear factors to 1e-6 is not certified; block stationarity and measured nonlinear convergence are reported instead.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C5_2_C5_3_C5_4_exact_blocks_and_stationarity` | PASS |
| `test_C5_5_C5_7_real_nonnegative_weights_KKT` | PASS |
| `test_C5_1_fixed_point_C5_8_frozen_gates_C5_13_no_retrace` | PASS |

| Measured quantity | Relative value | Requirement |
| --- | --- | --- |
| object_LS_gradient_ratio | 1.4219e-12 | < 1e-8 |
| probe_LS_gradient_ratio | 2.1203e-15 | < 1e-8 |

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
