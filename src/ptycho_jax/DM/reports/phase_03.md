# Phase 03 implementation report

Hermitian adjoints are independently checked with linear_transpose. The fractional object normal is non-diagonal; its exact diagonal is used only as a preconditioner. The probe normal is diagonal for all positions.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C3_1_E_dot_and_C3_5_coverage` | PASS |
| `test_C3_2_S_unitary_and_C3_3_L_dot` | PASS |
| `test_C3_3_C3_4_combined_hermitian_adjoints` | PASS |
| `test_C3_6_integer_object_C3_7_all_probe_normals_diagonal` | PASS |
| `test_C3_8_fractional_object_diagonal_is_exact_diagonal_not_normal` | PASS |
| `test_C3_9_C3_10_chunked_adjoint_and_nonnegative_diagonals` | PASS |

| Measured quantity | Relative value | Requirement |
| --- | --- | --- |
| object_dot_relative_error | 2.4373e-16 | < 1e-12 |
| object_linear_transpose_relative_error | 2.1607e-16 | < 1e-12 |
| probe_dot_relative_error | 1.1914e-16 | < 1e-12 |
| probe_linear_transpose_relative_error | 9.2068e-17 | < 1e-12 |

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
