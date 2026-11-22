# Phase 0: Audit the AD forward model (conventions sheet)

## Purpose

Every later phase depends on exact conventions: FFT scaling, shift sign, weight parametrization, loss definition. A single wrong sign or scale produces a DM that looks plausible but fails at hand-off. This phase writes those conventions down once, verifies them numerically, and saves golden reference data for regression tests.

## Inputs

- Path to the AD forward model and loss function, and to the optax driver.
- A small example dataset or the ability to generate random inputs of the right shapes.

## Tasks

### 0.1 Fill the conventions sheet (`docs/conventions.md`)

For each row give the value, the evidence (file:line), and notes.

| Topic | What to record |
|---|---|
| Shapes and dtypes | probe modes, object, weights, positions, data, masks |
| Object representation | complex vs (amplitude, phase); canvas size, padding, origin |
| Positions | units (pixels, metres, fraction), axis order (y,x), origin, **sign** (does a positive position move the window forward or backward in the object?) |
| How positions enter | integer crop, bilinear interpolation, Fourier shift, sinc; which of probe or object is shifted |
| Probe | mode axis position; normalized or not; shift convention |
| Weights | parametrization (raw, squared, softplus, softmax, log); normalized per position or globally; per-position or global; any separate per-position flux scale |
| FFT | normalization (ortho or not), fftshift/ifftshift placement, centered origin, axes |
| Data | intensities or amplitudes; units (photons); background or offset terms; detector gain |
| Loss | form (Poisson NLL, amplitude L2, intensity L2), reduction (sum or mean), mask handling, epsilons |
| Extra model terms | anything beyond `I_j = sum_k w_jk |F[S_j P_k * E_j O]|^2` |
| Numerics | JAX version, default dtype, x64 enabled or not |

### 0.2 Decide DM-equivalence

List every feature of the AD model that the DM model (README section 3) does not represent. For each one choose: ignore, approximate (state how), or **stop and ask the user**. Do not silently drop terms such as an additive background or a per-position flux factor.

### 0.3 Write `conventions.py`

Expose:
- `fft2c`, `ifft2c`: exactly the AD model's FFT convention (ortho strongly preferred, so Parseval holds).
- `weights_to_a`, `a_to_weights_param`: converters between the AD weight parametrization and the DM square-root form `a = sqrt(w)`.
- `split_positions`: positions to (integer part n, remainder delta), using the AD model's units and sign.

### 0.4 Golden data

Run the reference model on a small, seeded random case (for example J=9, K=2, 32x32 frames) with non-integer positions. Save inputs and outputs to `tests/golden/ref_small.npz`. Later phases compare against this file.

## Checks

| ID | Check | Tolerance |
|---|---|---|
| C0.1 | **Parseval per position.** For the reference model, `sum_q I_model_j` equals `sum_k w_jk * sum_q abs(shifted probe_k * object patch_j)^2` (with the correct shift handling) | tol64 |
| C0.2 | **Reproducibility.** Re-running with the same seed reproduces the golden file | exact |
| C0.3 | **Analytic toy.** One position, K=1, object = 1, Gaussian probe: model output equals `abs(F P)^2` | tol64 |
| C0.4 | **Shift sign.** Object with a single bright pixel; move the position by +1 pixel and confirm in which direction the bright spot moves inside the patch. Record this as the sign convention used by `E_j` and `S_j` | exact (documented) |
| C0.5 | **Weights round trip.** `a_to_weights_param(weights_to_a(x)) == x` for random valid x | tol64 |

## Gate

- `docs/conventions.md` is complete and has no unresolved "?".
- C0.1 to C0.5 pass.
- Every unrepresentable model term is resolved or has been raised with the user.

## Pitfalls

- If C0.1 fails, the FFT normalization or the weight convention is misread. Fix the sheet before continuing.
- Do not assume `fftshift` is applied. Read the code.
- Record whether the loss normalizes by the number of pixels or positions, because Phase 11 compares losses.
