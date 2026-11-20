# 3D Bragg ptychography forward model (JAX) — plan & progress

Branch `feature_3dBPP`. Theory: Maddali et al. 2020 (Part I), Li et al. 2020 (Part II); spec: [braggTODO.md](braggTODO.md).
Run tests: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests -q` (28 pass). Demo: `.venv/bin/python run_bragg_validation.py` -> `bragg_validation.png`.

## Model (as implemented)
Object lives on the **beam-aligned grid (u,v,w)** at the Bragg condition (w || k_f, (u,v) = detector plane); it is a pure rotation of the sample frame, so no shear is needed for the object. Bragg carrier removed.

Once per update (`probe_constructor`, differentiable):
2D envelope -> exact angular-spectrum stack along native axis w' (carrier removed) -> 3-shear Fourier rotation by 2θ_B -> `P_ref(u,v,w)`.

Per scan point (vmapped):
`floor` integer corner (`dynamic_slice`, w integer) -> edge taper -> sub-voxel Fourier shift in (u,v) = residual + `tanh(scan_mistakes)*max_corr` -> crop -> `P_ref * patch`
-> separable ramp `exp(+2πi ΔQ(θ)·r)` -> sum over w -> orthonormal 2D FFT -> |.|².
ΔQ(θ) = R_y(θ)Q0 − Q0 in the beam frame (exact, differentiable in θ). Conventions are documented in `bragg/geometry.py`.

## Files
| file | content |
|---|---|
| `bragg/geometry.py` | `BraggGeometry`, rotation, beam basis, Q(θ), ΔQ in beam frame, scan->grid positions |
| `bragg/probe.py` | `angular_spectrum_stack`, `shear_via_fourier`, `rotate_volume_fourier_shear`, `build_reference_probe` |
| `bragg/forward.py` | taper, Fourier shift, ramps, `detector_amplitude`, `make_bragg_model` (probe_constructor / point_model / batch_model; same signature style as the transmission models) |
| `bragg/reference.py` | analytic soft-cube phantom with slow phase, analytic Gaussian beam, **exact 3D DFT of probe×object at the detector q-points**, rotated-envelope truth |
| `tests/` | `test_geometry`, `test_probe`, `test_forward_vs_3d`, `test_batch_and_grad` |

## Plan / status
- [x] 1 Geometry + tests (orthonormal right-handed basis, Q0 ∥ z, |Q| conserved, ΔQ(0)=0, analytic first-order ΔQ via autodiff)
- [x] 2 Shear/rotation + tests (shear = exact shift; rotation vs analytic Gaussian incl. 90°; round trip; energy)
- [x] 3 Probe + tests (propagation vs analytic Gaussian beam 1e-4; full `build_reference_probe` vs analytic rotated beam 3e-3, truncation-limited by `extra_w`)
- [x] 4 Forward core + position handling
- [x] 5 **Main gate: model vs exact 3D calculation** (cube with slow phase)
  - θ=0 equals plane q_w=0 of `fftn(P·ρ)`: 1e-10
  - rocking θ within ±1.6 mrad vs exact 3D DFT of P·ρ at the detector-plane q-points: ~1e-15 (machine precision)
  - fractional scan position vs analytic object evaluated at that position: <2e-3; trainable correction reproduces a true sub-voxel offset: <2e-3; integer w offset exact
  - Parseval/normalisation holds
- [x] 6 Validity range of the phase-ramp approximation (object envelope physically rotated by θ in the exact calc): rel. error 6e-4 @0.25 mrad, 2.5e-3 @0.5, 8.8e-3 @1.0, 1.8e-2 @1.5 mrad (∝θ²)
- [x] 7 Batch/JIT/gradients: vmap == loop, probe built once per batch call, no re-jit, autodiff vs finite differences for object, probe_2d and scan_mistakes (<1e-5)
- [ ] 8 Integration into existing pipeline (`pipeline.py`, optimiser loop, loss; map `scan_mistakes` optimizer keys; load real data / `scan_to_grid_positions`)
- [ ] 9 Synthetic ptychographic reconstruction: simulate scan with perturbed positions, reconstruct object + positions with the existing optimiser, check `scan_mistakes` convergence
- [ ] 10 Clean up scratch `bragg_ptycho_models.py`/`bragg_models.py`; optional: orthogonalisation of the reconstructed object (Part II) as post-processing; float32 performance/memory profiling (chunked vmap)

## Notes / limitations
- Exact-3D comparison checks implementation (conventions, shifts, normalisation) to machine precision; the physical approximation (envelope not rotated, detector plane q_w=0, i.e. small-angle detector) is quantified in item 6. Valid θ range is also bounded by w sampling: |ΔQ_w| < 1/(2 dw) (here ≈ ±1.6 mrad).
- θ sign: sample rotation by +θ about v gives ΔQ_w ≈ +sin(2θ_B)θ/λ.
- The probe rotation angle is 2θ_B (k_i -> k_f), verified against the analytic Gaussian beam.
- The rotated native box does not cover the reference box: native volume is propagated over `Nw+2*extra_w` planes (default ceil(Nu·|sin2θ_B|) per side), then cropped. Increase `extra_w` for wide/strongly diverging beams.
- Position w-component is integer-voxel only; sub-voxel correction is in (u,v) (the object grid in w is set by the rocking range).
- Tests run in float64 (`jax_enable_x64`); the model itself is dtype-agnostic.
