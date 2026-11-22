# Phase 11 implementation report

The last overlap factors transfer exactly to the full complex AD matrix parameterization with balanced scale. Positions are fixed injected arrays and absent from optimizer leaves. Both nominal and +30% NanoMAX cases and equal-AD-budget controls were executed.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C11_1_to_C11_4_exact_handoff` | PASS |
| `test_C11_5_AD_descends_and_positions_frozen` | PASS |

| Measured quantity | Relative value | Requirement |
| --- | --- | --- |
| MSE handoff error | 3.1266e-15 | < 1e-12 |

| NanoMAX probe defocus | DM MSE | Handoff relative error | Conservative AD final | Original-rate AD final |
| --- | --- | --- | --- | --- |
| 1.500 mm | 10.2513 | 9.076e-08 | 3.47539 | 2.81624 |
| 1.950 mm | 133.255 | 1.562e-07 | 47.2621 | 36.5863 |

The +30% case remains visually poor and high-error; lower loss does not establish reliable specimen recovery. AD controls share rates/epochs but DM→AD has additional DM compute.

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
