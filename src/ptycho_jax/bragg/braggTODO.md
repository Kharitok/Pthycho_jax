TODO: Bragg Ptychography Forward Model (JAX) — Implementation & Testing Plan

This document specifies all functions needed to implement a phase-ramp-based forward model for Bragg ptychography, plus the testing suite to validate it. Organized by module and implementation order.

Conventions (fix these before coding)
Sample frame: X, Y in sample plane; Z ∥ Q at θ = 0; rocking axis = Y.
Reference angle θ_ref = 0 (nominal symmetric Bragg condition).
Beam frame axes: (u, v, w) with w ∥ k_f(θ_ref), (u,v) in detector plane.
Crystallographer’s Fourier convention: exp(−2πi r·q).
All physical pitches (du, dv, dw, dX, dY, dZ, dq) defined consistently; document units (meters, inverse meters) in a config/geometry object.
Module 1: geometry.py — Rotation & Bragg geometry primitives
Function	Signature	Purpose
rotation_matrix_y	(theta) -> (3,3)	Rotation matrix about Y
beam_basis	(theta, theta_B) -> (k_i, k_f, e1, e2)	Beam-frame unit vectors at angle θ, in sample frame
reference_basis_matrix	(theta_B) -> (3,3)	Fixed matrix [e1_0 | e2 | k_f0] at θ_ref=0
bragg_Q_vector	(theta, theta_B, wavelength) -> (3,)	Exact Q(θ) in sample frame
q_shift_in_ref_frame	(theta, theta_ref, theta_B, wavelength, M_ref) -> (3,)	ΔQ projected into reference beam frame

Status: [ ] Not started / [ ] In progress / [ ] Done

Module 2: grids.py — Static coordinate grids
Function	Signature	Purpose
build_beam_coordinate_grid	(Nu, Nv, Nw, du, dv, dw) -> (U, V, W)	Physical coords of every probe voxel
q_grid_from_object_grid	(Nx, Ny, Nz, dX, dY, dZ) -> (qx_axis, qy_axis, qz_axis)	Q-axes matching a 3D FFT of the object grid (test-only, but keep here for reuse)
detector_plane_in_sample_Q	(theta, theta_B, wavelength, u_axis, v_axis) -> (Nu,Nv,3)	Sample-frame Q-coords of every detector pixel at angle θ (test-only)

Status: [ ]

Module 3: propagation.py — Probe construction
Function	Signature	Purpose
angular_spectrum_propagate	(field0, wavelength, dx, dy, z) -> complex 2D	Propagate 2D field by z along native axis
build_probe_native_volume	(field0, wavelength, du, dv, w_native_list) -> (Nu,Nv,Nw)	Stack propagated field → native-frame 3D probe

Status: [ ]

Module 4: shear_rotation.py — Fourier shear-based 3D rotation
Function	Signature	Purpose
shear_via_fourier	(volume, shear_axis, dependent_axis, shear_factor, d_shear, d_dep) -> complex array	Exact per-line Fourier shift (one shear pass)
rotate_volume_fourier_shear	(volume, theta, du, dw, pad_u=0, pad_w=0) -> complex array	Full rotation via 3-pass shear decomposition
build_reference_probe	(field0, wavelength, theta_B, du, dv, w_native_list, pad_u=0, pad_w=0) -> P_ref	Full probe pipeline: native propagation + one rotation into reference frame

Status: [ ]

Implementation notes for agent:

Shear decomposition: R(θ) = Shear_u(tan(θ/2)) ∘ Shear_w(−sinθ) ∘ Shear_u(tan(θ/2)).
Rotation mixes only (u,w); v is a free batch axis — must broadcast correctly.
Zero-pad before rotating if shear amount approaches array half-width, to avoid circular wraparound; crop back after.
Confirm sign convention of rotation angle matches beam_basis’s R_y(-theta) usage.
Module 5: ramps.py — Rocking-angle phase ramps
Function	Signature	Purpose
rocking_phase_ramp	(dQ, U, V, W) -> complex array	Elementwise phase factor for one angle
precompute_all_ramps	(unique_thetas, theta_ref, theta_B, wavelength, M_ref, U, V, W) -> (N_theta, Nu,Nv,Nw)	Vectorized ramp cache over all unique angles

Status: [ ]

Module 6: forward_core.py — Per-scan-point forward pass
Function	Signature	Purpose
extract_sample_patch	(rho, X, Y, obj_origin, obj_voxel, patch_shape) -> (patch, residual_x, residual_y)	Integer-voxel slice + sub-voxel remainder
project_along_beam	(exit_volume, dw) -> complex 2D	Sum along w (projection operator ℛ)
subpixel_shift_2d	(field_2d, residual_x, residual_y, du, dv) -> complex 2D	Apply leftover (X,Y) offset as 2D Fourier shift
propagate_to_detector	(exit_2d, du, dv) -> complex 2D	2D FFT to detector field
detector_intensity	(psi) -> real 2D	|psi|^2
forward_single	(theta_idx, X, Y, rho, P_ref, ramps_cache, obj_origin, obj_voxel, du, dv, dw) -> I	Full composed forward pass, one scan point
forward_batch	jax.jit(jax.vmap(forward_single, ...))	Vectorized over full scan

Status: [ ]

Module 7: setup.py — Top-level orchestration (one-time init)
Function	Signature	Purpose
init_geometry	(theta_B, wavelength, p, D, N_det) -> dict	Compute du, dv, dq, M_ref
init_probe	(field0, w_native_list, geometry) -> P_ref	Call build_reference_probe
init_ramps	(all_thetas, theta_B, wavelength, M_ref, geometry) -> (ramps_cache, theta_idx_array)	Call precompute_all_ramps, build lookup table from scan→cache index

Status: [ ]

TESTING SUITE
Test Module A: test_geometry.py — Level 1 unit tests
 test_rotation_identity: R_y(0) == I
 test_beam_basis_orthogonality: at θ=0, k_f0 · e1_0 ≈ 0, |k_f0| == |k_i0| == 1
 test_Q0_magnitude_direction: Q(0) ∥ Z, |Q(0)| == 2 sin(theta_B)/wavelength
 test_Q_small_angle_limit: finite-difference derivative of bragg_Q_vector at θ=0 matches analytic first-order term
 test_q_shift_zero_at_theta_ref: q_shift_in_ref_frame(theta_ref, theta_ref, ...) == 0
Test Module B: test_shear_rotation.py — Level 1 unit tests
 test_shear_is_shift: 1D shear of a Gaussian matches known sub-pixel-shifted reference to high precision
 test_shear_energy_conservation: sum(|shifted|^2) == sum(|original|^2) (within padding tolerance)
 test_rotation_vs_scipy: rotate_volume_fourier_shear vs scipy.ndimage.rotate for same angle — close agreement
 test_rotation_roundtrip: rotate by θ then −θ → recovers original (checks sign convention)
 test_rotation_90deg: known axis-swap/flip behavior for θ=90° (catches axis-order bugs round-trip can’t)
Test Module C: test_ramp_vs_direct_rotation.py — Level 1/3 cross-check
 test_ramp_matches_direct_small_theta: P_ref * ramp(θ) projected along w vs. rotate_volume_fourier_shear(P_native, theta_B+θ) projected along w — agree for small θ, diverge for large θ (plot error vs θ to find valid range)
Test Module D: test_3dfft_ground_truth.py — Level 2 core validation (your main requested test)

Setup:

 build_test_cube(...): synthetic object (uniform box, optionally with linear phase gradient for strain test)
 compute_3d_far_field(rho, dX, dY, dZ): ground-truth 3D FFT
 extract_slice_from_3d_fft(Psi_3D, qx_axis, qy_axis, qz_axis, Q_points): interpolate 3D Fourier volume at arbitrary Q

Tests:

 test_single_angle_match: at small θ_test, flat-probe framework output vs. 3D-FFT slice — relative error below threshold (e.g. 1e-3)
 test_angle_sweep_error_curve: sweep θ from 0 to beyond expected valid range; plot/assert error stays low in-range and grows outside it
 test_visual_fringe_check: side-by-side image comparison (manual/visual regression, store reference images) — catches mirrored/skewed/rotated fringes from sign or axis bugs
 test_phase_gradient_object: repeat with a cube carrying a linear phase ramp (simulated displacement) — validates phase/displacement sign convention, not just amplitude
Test Module E: test_integration.py — Level 3 incremental re-complexification
 test_real_probe_windowing: replace flat probe with real P_ref; compare against 3D-FFT of probe_window(θ_ref) * rho_test
 test_nonzero_XY_offset: shift object/probe by known (X,Y); compare forward_single output against correspondingly-shifted ground truth
 test_energy_normalization: Parseval check — sum(|psi|^2) vs sum(|exit_2d|^2) scaling, catches missing/extra du*dv, dw factors
 test_theta_equals_theta_ref: ramp at θ=θ_ref is exactly 1 everywhere; forward pass reduces to “no ramp” case
Test Module F: test_batch_and_jit.py — Infrastructure checks
 test_vmap_matches_loop: forward_batch over N points equals a plain Python loop calling forward_single N times
 test_jit_no_recompile: confirm forward_batch doesn’t retrace/recompile across calls with same shapes (check with jax.jit cache stats or timing)
 test_gradient_flows: jax.grad w.r.t. rho through forward_single/forward_batch produces finite, non-NaN gradients (needed for later phase retrieval)
Suggested implementation order for the coding agent
Module 1 (geometry.py) + Test Module A — get geometry right before anything else depends on it.
Module 2 (grids.py) — trivial, no tests needed beyond shape checks.
Module 4 (shear_rotation.py) + Test Module B — validate rotation in isolation before building the probe pipeline on top of it.
Module 3 (propagation.py) — straightforward, can add a basic “probe stays normalized” sanity check.
Module 4’s build_reference_probe (ties 3+4 together).
Module 5 (ramps.py).
Test Module C — cross-check ramp vs. direct rotation, now that both exist.
Module 6 (forward_core.py).
Test Module D — the full 3D-FFT ground truth test; this is the main correctness gate.
Test Module E — reintroduce probe windowing, offsets, normalization checks.
Module 7 (setup.py) — orchestration, once all pieces are validated individually.
Test Module F — batching/JIT/gradient infrastructure checks, last, since they depend on everything above being correct and stable.

Gate criterion before moving to real experimental data: Test Module D’s angle-sweep error curve must show a clear, documented valid angular range with low error, and Test Module E’s offset/normalization tests must all pass.