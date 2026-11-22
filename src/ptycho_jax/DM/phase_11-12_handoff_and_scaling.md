# Phases 11 and 12: Hand-off to optax, memory and multi-device scaling

---

## Phase 11: Hand-off to the optax stage

### Purpose

Convert the DM result into the exact parameterization of the AD model, verify that the AD loss at those parameters equals the DM model error, and show that optax starts from a better point than a naive initialization.

### Steps

1. **Take the right quantities.** Use `(probe, obj, a)` from the **last overlap projection** (the current state), not raw `psi`. Apply the gauge normalization of Phase 6 once more at the end.
2. **Convert to the AD parameterization** (`handoff.py`), following `docs/conventions.md`:
   - Object: complex, or amplitude and phase, as the AD model expects. Match canvas size and padding.
   - Probe modes: match shape and mode order. The DM modes have unit norm; the AD model may carry probe power in the modes or in the weights, so move the scale accordingly.
   - Weights: `w = a^2`, then apply the inverse of the AD parametrization (for example `log(expm1(w))` for softplus, logits for softmax plus a separate scale for normalized weights). If the AD model has a separate per-position flux factor, derive it consistently (`flux_j = sum_k w_jk`) and normalize the weights per position.
   - Intensity scale and FFT normalization: as in the conventions sheet.
3. **Verify** the AD loss at the converted parameters (below).
4. **Short optax trial:** 5 to 10 steps with the AD pipeline's optimizer and a conservative learning rate (the pipeline's usual value divided by about 10), probe, object and weights free, **positions frozen**.

### Checks

| ID | Check | Tolerance |
|---|---|---|
| C11.1 | The AD loss at the converted parameters is finite | exact |
| C11.2 | **The AD loss at the converted parameters equals the DM model error M3** (same units) | 1e-4 relative (float32), tol64 (float64) |
| C11.3 | **Round trip:** convert DM to AD parameters and back; the model intensities are identical | tol64 |
| C11.4 | Weights respect the AD model's constraints (non-negative; normalized if required) | exact |
| C11.5 | The optax trial's loss decreases without blow-up (small non-monotonic noise is allowed), and is **lower than a naive-initialization start at equal step counts** | qualitative plus recorded numbers |
| C11.6 | Any AD-side regularizers or constraints (probe norm, object bounds) are satisfied at the converted parameters, or the conversion adjusts them | exact |
| C11.7 | Hand-off on synthetic cases A to F: loss after 50 optax steps from DM start vs naive start is reported in the ladder report | reported |

### Gate

C11.1 to C11.6 pass.

### Pitfalls

- A mismatch in C11.2 almost always means a convention error (FFT normalization, weight parametrization, shift sign). Return to Phase 0 and the golden file rather than adjusting tolerances.
- Do not hand off the object as it stands if the AD model's amplitude/phase representation wraps the phase differently: check at the object borders.

---

## Phase 12: Memory and performance (do this after correctness is established)

### Memory model

`psi` holds `J * K * Ny * Nx` complex values. In complex64 that is `8 bytes * J * K * Ny * Nx`. Example: 10,000 positions x 4 modes x 256 x 256 is about 21 GB for one copy. A naive step holds several copies at once (`psi`, `psi_F`, `psi_in`, `psi_O`, the shifted-probe array, FFT temporaries), so peak memory can be 5 to 8 times that. Measure it rather than guessing.

### Mitigations, in order of preference

1. **complex64 / float32** for production.
2. **Fewer modes in DM** (for example K = 1 or 2); let the optax stage add more.
3. **Chunking over positions** with `lax.map` (batch size) or `scan` over chunks with an accumulator carry. P_F and per-position weight updates are chunk-local. The object and probe numerators and denominators are **sums over positions** and are accumulated in the carry. Because the inner loop of P_O reuses `psi_in`, one full copy of `psi_in` must be stored (or recomputed per inner iteration, trading time for memory); measure both.
4. **Donation** of `psi` (`donate_argnums`).
5. **Sharding over positions** (`shard_map` or `jax.sharding` with a `Mesh` axis for positions): `psi` and the per-position arrays are sharded; object and probe are replicated; the sums in P_O finish with a `psum` across the position axis. P_F needs no communication.
6. **Cache the shifted probes** `S_j P_k` within an inner iteration only if memory allows, and recompute after each probe update.

### Checks

| ID | Check | Tolerance |
|---|---|---|
| C12.1 | Chunked step equals unchunked step | 1e-5 relative (float32), tol64 (float64) |
| C12.2 | Sharded step equals single-device step. Simulate devices on CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=4` | 1e-5 relative (float32) |
| C12.3 | **Benchmark table** for at least three problem sizes: time per iteration, peak memory, with and without chunking, with `diag` and `cg` probe solvers, with `integer_positions_only` on and off | reported |
| C12.4 | No recompilation across iterations or across schedule boundaries in the chunked and sharded variants | trace count = 1 |
| C12.5 | Profile one iteration (`jax.profiler`) and list the three most expensive operations with the share of time each | reported |

### Gate

C12.1 to C12.4 pass and the benchmark table is in `reports/phase_12.md`.

### Deliverables of the whole project (final checklist)

1. Modules listed in README section 7.
2. One test file per phase, with test names carrying the check IDs.
3. `reports/phase_XX.md` for every phase.
4. `docs/conventions.md` and `docs/decisions.md`.
5. `scripts/run_validation_ladder.py`, its results table and plots.
6. A short usage example: run DM on a dataset, hand off to optax.
