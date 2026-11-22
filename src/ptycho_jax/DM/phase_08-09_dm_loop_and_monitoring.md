# Phases 8 and 9: DM step, schedule, loops, monitoring

---

## Phase 8: The DM iteration

### State and step

State pytree: `DMState(psi, probe, obj, a, it)`.

```
dm_step(state, data, geom, cfg) -> (state, metrics)      # pure, jittable
```

### beta = 1 step (implement first)

1. `psi_F, fstats = P_F(psi)`
2. `psi_in = 2 * psi_F - psi`
3. `psi_O, probe, obj, a = P_O(psi_in; warm start from state; gates from state.it)`
4. `psi_new = psi + psi_O - psi_F`
5. Gated housekeeping on the new state: mode orthogonalization (if due), collapse guard, `it += 1`.
6. Assemble metrics (Phase 9).

### General beta (implement after beta = 1 passes)

With `beta != 1`:

```
f_F(psi) = (1 + 1/beta) * P_F(psi) - psi / beta
f_O(psi) = (1 - 1/beta) * P_O(psi) + psi / beta
psi_new  = psi + beta * ( P_O(f_F(psi)) - P_F(f_O(psi)) )
```

This needs **two P_F and two P_O calls** per iteration. Only the `P_O(f_F(psi))` call updates the persistent (P, O, a) in the state. The second call, `P_O(psi)`, is **stateless**: warm-start from the same state, compute `psi_O`, discard the updated (P, O, a). At `beta = 1` this reduces exactly to the step above (`f_O(psi) = psi`, so `P_F(f_O(psi)) = P_F(psi)`, which is reused).

### Schedule (all gates are traced, derived from `state.it`)

| Stage | Condition (defaults) |
|---|---|
| Object only | `it < n_probe_start` (10) |
| + probe modes | `it >= n_probe_start` |
| + weights, `global` | `it >= n_weight_global_start` (30) |
| + weights, `per_position` | `it >= n_weight_pp_start` (50) |
| Orthogonalization | `n_ortho > 0 and it % n_ortho == 0 and it >= n_probe_start` |
| Object gauge | every iteration |

`weight_mode = 0/1/2` is computed with `jnp.where` from `it`. Total length is about 100 to 300 iterations.

### Loops

- `run_dm(state, data, geom, cfg, n_iter)`: `jax.lax.scan` over a **block** of `log_every` iterations, recording the stacked metrics. A Python loop calls the jitted block repeatedly. Between blocks the Python side may log, save snapshots, check for NaN, and plot.
- Use `donate_argnums` for the state so psi is not duplicated.
- A fully-scanned run over all iterations is allowed when no live feedback is needed.

### Checks

| ID | Check | Tolerance |
|---|---|---|
| C8.1 | **Fixed point:** start from psi generated from the true (P, O, a) with noise-free data and consistent state; one step returns psi unchanged | 1e-6 (float32), 1e-10 (float64) |
| C8.2 | **Compile once:** a Python counter inside the traced step shows exactly one trace across many steps, including steps that cross every schedule boundary | trace count = 1 |
| C8.3 | `lax.scan` block and the equivalent Python loop of single steps produce identical results (float64) | 1e-12 |
| C8.4 | **Independent reference for the reduced case:** with `weight_mode = fixed`, K = 1, integer positions, compare the full JAX DM against a separate, simple numpy implementation of single-mode DM written without the JAX operators (small problem, for example J = 25, 32x32 frames, 20 iterations). It guards against shared conjugation or sign errors | 1e-10 |
| C8.5 | General-beta code path at `beta = 1` equals the specialized beta = 1 step | 1e-12 |
| C8.6 | General beta with `beta != 1`: stateless second P_O leaves the persistent state identical to what one P_O call would give | exact |
| C8.7 | Donation works: no "donated buffers were not usable" warning; the input state is not reused after the step | exact |
| C8.8 | After each block, all arrays are finite; on NaN the loop stops and saves a debug dump (last good state, config) | exact |
| C8.9 | Schedule gates behave: before `n_probe_start` the probe is unchanged; before `n_weight_global_start` the weights are unchanged; before `n_weight_pp_start` the weights are position-independent | exact |

### Gate

C8.1 to C8.9 pass.

---

## Phase 9: Monitoring

### Metrics recorded per iteration (or every `eval_every` for the costlier ones)

| ID | Name | Definition | Cost |
|---|---|---|---|
| M1 | Fourier error | `sum_valid (sqrt(I_model(psi)) - sqrt_I)^2 / sum_valid I_meas`, the iterate's modulus error, from P_F | free |
| M2 | DM update norm | `norm(psi_O - psi_F)^2 / norm(psi_F)^2` (for beta = 1 equals the squared step size). The standard DM convergence indicator | free |
| M3 | **Model error** | The Phase 1 stage model (or the reference AD model) evaluated on the current `(probe, obj, a)`, compared with the data, in the **same units as the AD loss** and also as `R = sum_valid (sqrt(I_model) - sqrt_I)^2 / sum_valid I_meas`. This is the number that matters for the hand-off | one extra FFT pass, so every `eval_every` |
| M4 | Weight statistics | per mode: mean, std over positions, min; fraction of clamped (zero) weights; collapse-guard events | cheap |
| M5 | Norms | `norm(obj)` over the covered region, probe mode norms (1 after normalization) | cheap |
| M6 | Optional diagnostic | overlap residual `norm(psi_F - P_O(psi_F))^2` every N iterations (needs an extra P_O call) | extra P_O |

Also write snapshots (object amplitude and phase, probe modes, weights versus position) every `snapshot_every` iterations.

### Checks

| ID | Check | Tolerance |
|---|---|---|
| C9.1 | M1 equals an independent numpy computation of the Fourier error | tol32 |
| C9.2 | **M3 equals the AD loss** from the reference model evaluated on the same parameters (after conversion) | 1e-4 relative (float32), tol64 (float64) |
| C9.3 | M2 equals `norm(psi_new - psi)^2 / norm(psi_F)^2` for beta = 1 | tol64 |
| C9.4 | Metrics stack correctly from `lax.scan` (shape `[n_block]` per metric) | exact |
| C9.5 | On a synthetic run, M3 trends downward and plateaus; M1 and M2 fluctuate (this is normal for DM and is **not** a bug) | qualitative |

If C9.2 fails there is a convention mismatch. Return to Phase 0.

### Gate

C9.1 to C9.4 pass.
