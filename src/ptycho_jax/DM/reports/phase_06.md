# Phase 06 implementation report

Compensated object/mode gauges preserve fields. Inactive zero-weight modes freeze and are logged. Optional constant-weight mode canonicalization and the non-exact per-shot rotation heuristic are not implemented or claimed as tested. Production DM orthogonalization is off.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C6_1_C6_2_C6_3_gauge_preserves_fields` | PASS |
| `test_C6_6_zero_mode_freezes_and_is_logged` | PASS |

| Measured quantity | Relative value | Requirement |
| --- | --- | --- |
| Gauge field error | 3.4809e-16 | < 1e-12 |

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
