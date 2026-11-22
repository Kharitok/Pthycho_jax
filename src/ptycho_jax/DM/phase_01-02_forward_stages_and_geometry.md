# Phases 1 and 2: Forward-model stages and fixed-position geometry

## Purpose

DM needs the forward model in pieces, because it works on the exit waves between the pieces. This document splits the model into stages (Phase 1) and sets up the fixed-position geometry that the DM operators use (Phase 2). Positions never change.

---

## Phase 1: Split the forward model into stages

### Functions (all pure, jittable, shapes in docstrings)

| Function | Signature | Meaning |
|---|---|---|
| `extract_patches` | `(obj, geom) -> [J, Ny, Nx]` | E: integer-pixel crop of the canvas at n_j |
| `shift_probe` | `(probe, geom) -> [J, K, Ny, Nx]` | S_j P_k: Fourier phase-ramp shift of each mode by delta_j |
| `exit_waves` | `(probe, obj, a, geom) -> [J, K, Ny, Nx]` | `a[:, :, None, None] * shift_probe * patches[:, None]` |
| `propagate` | `(psi) -> [J, K, Ny, Nx]` | `fft2c` over the last two axes |
| `detector_intensity` | `(Fpsi) -> [J, Ny, Nx]` | sum over modes of abs squared |
| `forward_from_stages` | `(probe, obj, a, geom) -> [J, Ny, Nx]` | composition of the above |

Keep the original AD model untouched as the reference. `forward_from_stages` must accept parameters in the DM form (`a`, not the AD weight parametrization); use the converters from `conventions.py` in tests.

### Checks

| ID | Check | Tolerance |
|---|---|---|
| C1.1 | `forward_from_stages` equals the reference model on the golden case and on random cases (after parameter conversion) | tol64 in float64, tol32 in float32 |
| C1.2 | **Gradient equivalence.** `jax.grad` of the AD loss through `forward_from_stages` equals the gradient through the reference model, for probe, object and weights (chain the weight converter). This catches conjugation and shift-sign slips that value checks can miss | tol64 / tol32 |
| C1.3 | All stage functions work under `jax.jit` and under `jax.vmap` over a leading batch axis where applicable | runs, same values |
| C1.4 | Dtype stability: float32/complex64 in gives float32/complex64 out; no silent promotion to 64-bit | exact |

Note: C1.1 and C1.2 hold exactly only if the reference model uses the same shift formulation. Phase 2 handles the case where it does not.

### Gate

C1.1 to C1.4 pass (or the Phase 2 mismatch decision below has been made and recorded).

---

## Phase 2: Geometry for fixed positions

### Tasks

1. **Split positions** once, outside jit: `n_j = round(x_j)` (or floor, but record the choice), `delta_j = x_j - n_j`. With rounding, delta lies in [-0.5, 0.5].
2. **Validity check** (outside jit, as an assertion): every crop `[n_j, n_j + (Ny, Nx))` lies inside the canvas. Pad the canvas if needed.
3. **Do not precompute per-pixel index arrays** of shape [J, Ny, Nx]. At J=10k and 256x256 frames each such int32 array is about 2.6 GB. Build indices on the fly from `n_j` plus `arange` by broadcasting (XLA fuses this), or use `jax.vmap(lax.dynamic_slice)` for the gather. For the scatter-add, benchmark `canvas.at[iy, ix].add(patches)` against a `lax.scan` of `dynamic_update_slice` additions; pick the one that is faster and lower in memory, and record the choice.
4. **Phase ramps** are computed on the fly from `delta_j`:
   `ramp_j(qy, qx) = exp(-2*pi*i*(qy*delta_y + qx*delta_x))`, with `q = fftfreq(N)` in the same layout (shifted or not) as the AD model's FFT convention. The sign is fixed by check C0.4.
5. `shift_probe`: take `fft2(P_k)` once per mode ([K, Ny, Nx]), multiply by the per-position ramp, inverse FFT. This costs J*K inverse FFTs per application. Provide a flag `integer_positions_only`: when all `delta` are 0, `S` is the identity and these FFTs are skipped.
6. **Wrap-around:** the Fourier shift is circular. The probe must be localized away from the frame edge (or the AD model has the same wrap behaviour). Document this.
7. Prepare data: `sqrt_I` [J,Ny,Nx], `mask`, per-position `counts = sum_q (mask * I_meas)`.

### Mismatch decision (only if the AD model interpolates the object)

If the AD model shifts the object by interpolation instead of shifting the probe by phase ramp, intensities differ slightly between the DM stages and the AD model. Quantify it with C2.1.
- If the relative intensity difference is below about 1e-3: accept and record.
- If larger: **stop and ask the user.** Preferred option: switch the AD model's shift to the same probe-phase-ramp formulation so DM and AD optimize the identical model. Do not alter the AD model without the user's approval.

### Checks

| ID | Check | Tolerance |
|---|---|---|
| C2.1 | With random P, O, a and **non-integer** positions, intensities from the stage formulation match the reference model | tol64 if same formulation; otherwise report the relative difference and apply the mismatch decision above |
| C2.2 | With **integer** positions (delta = 0), match is exact | tol64 |
| C2.3 | All crops are inside the canvas; coverage count map has minimum >= 1 inside the scanned region | exact |
| C2.4 | `jax.make_jaxpr` (or compiled memory analysis) of `extract_patches` and its scatter shows no [J,Ny,Nx]-sized constants | inspected, recorded |
| C2.5 | Round trip: `n_j + delta_j` reproduces the input positions | tol64 |

### Gate

C2.1 to C2.5 pass and the mismatch decision (if applicable) is recorded in `docs/decisions.md`.
