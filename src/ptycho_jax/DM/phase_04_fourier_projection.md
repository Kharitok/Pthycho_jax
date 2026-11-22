# Phase 4: Fourier (modulus) projection P_F

## Purpose

Project the exit waves onto the set of waves whose far-field mixed-state intensity matches the measurement. It acts independently at each scan position, so it is simple to batch in JAX.

## Function

```
fourier_projection(psi, sqrt_I, mask, eps) -> (psi_F, stats)
```

- `psi`: [J, K, Ny, Nx] complex
- `sqrt_I`: [J, Ny, Nx] measured amplitude
- `mask`: [J, Ny, Nx] bool, True = valid pixel
- `stats`: contains the Fourier error (see below), computed for free from the intermediate quantities

## Algorithm (per position, all modes together)

1. `Fpsi = fft2c(psi)` over the last two axes.
2. `I_model = sum_k abs(Fpsi_k)^2`.
3. Common rescale factor `r = sqrt_I / (sqrt(I_model) + eps)`.
4. Where `mask` is False: `r = 1`, i.e. leave the Fourier values unchanged.
5. `psi_F = ifft2c(r[:, None] * Fpsi)`.

All modes at a pixel are multiplied by the **same real factor**. This enforces the incoherent sum while preserving the relative modal amplitudes and phases at every pixel.

**Fourier error** (monitoring, from step 2): `sum_{valid} (sqrt(I_model) - sqrt_I)^2 / sum_{valid} I_meas`.

## Implementation notes

- Pure and jittable. Native batching over J; wrap in `lax.map(..., batch_size=...)` or a chunked `scan` when memory requires (Phase 12).
- `eps` is relative to the data scale (see README hyperparameters). Where `I_model` is about 0, `r` can be large but multiplies values that are about 0, so no NaN or Inf appears. Verify this.
- Use the same `fft2c` / `ifft2c` as the AD model (ortho), so the norm of `psi_F` equals the data norm on valid pixels.
- Do **not** add a relaxed or noise-aware modulus constraint in this version.

## Checks

| ID | Check | Tolerance |
|---|---|---|
| C4.1 | **Idempotence:** `P_F(P_F(psi)) == P_F(psi)` | 1e-6 (float32), tol64 (float64, with eps tiny) |
| C4.2 | **Constraint satisfied:** after P_F, `sum_k abs(F psi_F_k)^2 == I_meas` on valid pixels | 1e-6 relative |
| C4.3 | **Modal structure preserved:** ratios and relative phases between modes at each valid pixel in Fourier space are unchanged | tol64 |
| C4.4 | **Masked pixels untouched** in Fourier space | exact |
| C4.5 | **No NaN/Inf** with zero-count pixels, all-zero positions, and psi containing exact zeros | exact |
| C4.6 | **Fixed point:** psi generated noise-free by the stage model from known (P, O, a) satisfies `P_F(psi) == psi` | 1e-6 (eps tiny) |
| C4.7 | **Unitary mode mixing commutes:** for a random unitary U acting on the mode axis, `P_F(U psi) == U P_F(psi)`. This is needed by mode orthogonalization in Phase 6 | tol64 |
| C4.8 | **Norm identity:** `sum abs(psi_F)^2 == sum_valid I_meas + norm of the unchanged masked Fourier part` (Parseval with ortho FFT) | tol64 |
| C4.9 | Chunked evaluation equals unchunked; output dtype equals input dtype | tol64 / exact |

## Gate

C4.1 to C4.9 pass.

## Pitfalls

- Applying the correction per mode instead of with the common factor (this breaks mixed-state physics).
- Using intensities where amplitudes are required: `r` is an amplitude ratio.
- Forgetting the mask handling, which silently turns bad pixels into hard constraints.
