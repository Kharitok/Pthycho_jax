# Phase 04 implementation report

A shared multimode amplitude rescaling preserves masked Fourier values. Zero predicted amplitudes use a deterministic first-channel seed. This singular projection choice is documented and no noise relaxation is implemented.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C4_1_to_C4_4_C4_8_projection` | PASS |
| `test_C4_5_zeros_C4_6_fixed_C4_7_unitary` | PASS |

| Measured quantity | Relative value | Requirement |
| --- | --- | --- |
| PF_idempotence_relative_error | 2.6117e-16 | < 1e-12 |
| PF_intensity_relative_error | 3.4939e-16 | < 1e-12 |

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
