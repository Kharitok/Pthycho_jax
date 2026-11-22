# Phase 3: Linear operators, Hermitian adjoints, denominators

## Purpose

The overlap projection (Phase 5) is a set of closed-form least-squares solves. Each needs the adjoint of the exit-wave operator with respect to the object or the probe, and a diagonal denominator. This phase builds and verifies those pieces. A conjugation error here is the single most likely bug in the whole project, so the checks are strict.

## Complex adjoint convention

`jax.linear_transpose(f, example)` returns the **plain transpose** (no conjugation). For a complex-linear `f`, the Hermitian adjoint is

```
adjoint(f)(y) = conj( linear_transpose(f, example)(conj(y)) )
```

Write this once as `hermitian_adjoint(f, example_input)` in `conventions.py`. It is used as the **independent reference** in tests. Production code may use hand-written adjoints if they pass the checks and are faster.

## Operators (all complex-linear in their argument)

| Operator | Maps | Adjoint | Notes |
|---|---|---|---|
| `E` | canvas [Hc,Wc] -> patches [J,Ny,Nx] | `E_H`: scatter-add of patches into the canvas | E is a real 0/1 matrix, so E^T = E^H and `linear_transpose` of the gather is exactly `E_H` |
| `S` | modes [K,Ny,Nx] -> [J,K,Ny,Nx] | `S_H`: multiply by the **conjugate** ramp in Fourier space, per position | S is unitary per position; implement `S_H` explicitly |
| `A_O` (at fixed P, a) | canvas -> psi-shaped | `A_O_H(psi) = E_H[ sum_k a_jk * conj(S_j P_k) * psi_jk ]` | numerator of the object update |
| `A_P` (at fixed O, a) | modes [K,Ny,Nx] -> psi-shaped | `A_P_H(psi)_k = sum_j a_jk * S_j^H[ conj(E_j O) * psi_jk ]` | numerator of the probe update |

`A_O(O) = a * S P * (E O)` and `A_P(P) = a * S P * (E O)`, viewed as linear in O and in P respectively.

**Cost note:** `sum_j S_j^H[x_j]` can be computed as `ifft2( sum_j conj(ramp_j) * fft2(x_j) )`, needing J*K forward FFTs but only K inverse FFTs, which is about half the cost of applying `S_H` per position and then summing.

## Denominators (real, non-negative maps)

- **Object side** (canvas shape): `D_O = E_H[ sum_k a_jk^2 * abs(S_j P_k)^2 ]`
- **Probe side** (per mode): `D_P_k = sum_j a_jk^2 * shift_back_j( abs(E_j O)^2 )`, where `shift_back_j` applies the **inverse** of S_j's sub-pixel shift to the real map. Clip the result to >= 0, because Fourier shifting a real non-negative map can produce small negative ringing.

**Exactness:**
- `D_O` makes the object normal equations **exactly diagonal** (E is a pure crop, so `A_O^H A_O` is pointwise multiplication by `D_O`).
- `D_P` is exact for integer positions. For sub-pixel positions the true normal operator `sum_j a^2 S_j^H diag(abs(E_jO)^2) S_j` is not diagonal, and `D_P` is the standard diagonal approximation. An optional CG solver for the probe (Phase 5) removes the approximation.

## Tasks

1. Implement `E`, `E_H`, `S`, `S_H`, `A_O`, `A_O_H`, `A_P`, `A_P_H`, `D_O`, `D_P`.
2. Implement `hermitian_adjoint` and use it only in tests and as a fallback.
3. All operators batched and chunkable over J (`lax.map` / `scan` with a carry for sums).

## Checks (float64, random complex inputs)

| ID | Check | Tolerance |
|---|---|---|
| C3.1 | **Dot test for E.** `abs(<E x, y> - <x, E_H y>) / (norm(E x) * norm(y))` | < 1e-12 |
| C3.2 | **Dot test for S** (with non-zero delta) | < 1e-12 |
| C3.3 | **Dot test for A_O and A_P** (combined operators, with random a) | < 1e-12 |
| C3.4 | Hand-written `A_O_H`, `A_P_H` equal `hermitian_adjoint(A_O)`, `hermitian_adjoint(A_P)` | tol64 |
| C3.5 | `E_H(ones patches)` equals the coverage count map computed by an independent numpy loop | exact |
| C3.6 | **Diagonal claim, object:** `A_O_H(A_O(O)) == D_O * O` for random O (any positions, including sub-pixel) | tol64 |
| C3.7 | **Diagonal claim, probe, integer positions:** `A_P_H(A_P(P)) == D_P * P` | tol64 |
| C3.8 | **Probe approximation quality, sub-pixel positions** (informational): relative error between `A_P_H(A_P(P))` and `D_P * P`. Report it; if it is above about 5e-2, plan to use the CG probe solver | reported |
| C3.9 | `D_O >= 0` and `D_P >= 0` everywhere | exact |
| C3.10 | Chunked evaluation of every sum equals unchunked | tol64 |

## Gate

C3.1 to C3.7, C3.9, C3.10 pass. C3.8 is recorded in the report and the probe-solver choice (`diag` or `cg`) is written to `docs/decisions.md`.

## Pitfalls

- Forgetting `conj` in the numerators (the dot tests catch this).
- Using `linear_transpose` on a function that is not complex-linear (for example one containing `conj` or `abs`). Keep operators strictly linear.
- Applying `S_H` per position on a large array and then summing, when the Fourier-domain sum is cheaper (see cost note).
