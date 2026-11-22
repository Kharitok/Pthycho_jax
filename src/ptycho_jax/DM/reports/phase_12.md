# Phase 12 implementation report

Full versus chunked steps and schedule traces are tested. Four virtual CPU devices reproduce single-device results. Production NanoMAX uses one GPU because 169 shots are not divisible by two; no measurements are silently dropped. Profiling-operation percentage attribution is not implemented.

Current package suite: 37 tests, 0 failures. Detailed measurements: [JSON](measurements.json); [JUnit](tests.xml).

| Implemented check/test | Status |
| --- | --- |
| `test_C12_1_chunked_full_step` | PASS |

# DM compile/memory benchmark

One inner sweep, three CG iterations. Same three-mode problem sizes. The memory estimate is compiled arguments + outputs + temporaries − aliases, not an independently sampled process peak. Chunking bounds the backprojection intermediates; the complete DM iterate is still resident.

| Frame | Shots | Integer | Adjoint chunk | Seconds/iteration | Compiled buffers (MiB) |
| --- | --- | --- | --- | --- | --- |
| 16 | 16 | False | 0 | 0.000466143 | 0.886 |
| 16 | 16 | False | 8 | 0.000561139 | 0.714 |
| 16 | 16 | True | 0 | 0.000145821 | 0.620 |
| 16 | 16 | True | 8 | 0.000147936 | 0.620 |
| 32 | 25 | False | 0 | 0.000501434 | 3.958 |
| 32 | 25 | False | 8 | 0.000763497 | 3.254 |
| 32 | 25 | True | 0 | 0.000157978 | 2.582 |
| 32 | 25 | True | 8 | 0.000172911 | 2.582 |
| 64 | 49 | False | 0 | 0.000677097 | 23.988 |
| 64 | 49 | False | 8 | 0.00118465 | 22.157 |
| 64 | 49 | True | 0 | 0.00017836 | 19.779 |
| 64 | 49 | True | 8 | 0.000201449 | 19.779 |


Four-device relative errors: `{"psi": 1.265755644377457e-15, "probe": 7.53806512302316e-16, "obj": 8.592682783573653e-15, "a": 1.243647831842182e-15, "it": 0.0}`.

The 169×2×512×512 complex64 iterate is 676 MiB. Sampled production process usage was about 9428 MiB, including cached compiled workspaces. This sample is not a measured peak. Full/chunked variants retain the full iterate; the current chunking only bounds backprojection intermediates. The alternate diagonal-versus-CG probe benchmark in the guide is inapplicable here: the exact-compatible object-shift model has a diagonal probe solve and a CG object solve.

All departures and numerical settings are documented in [decisions](../docs/decisions.md).
