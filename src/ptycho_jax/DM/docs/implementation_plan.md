# Exact fixed-position difference-map implementation plan

**Goal:** Implement two-mode DM with per-shot nonnegative amplitude weights, exactly matching the existing transmission model, then compare DM-to-AD and AD-only NanoMAX runs for 1.50 mm and 1.95 mm probe initialization.

**Architecture:** Pure JAX operators and pytree state in this package. Keep the reference forward model and all previous pipelines untouched. Use its integer extraction, tapered enlarged object patch, negative-fraction Fourier shift and central crop exactly; use a preconditioned CG object block and an exact diagonal probe block. No gradient descent occurs in DM. AD is used only for independent algebra tests and the subsequent optax continuation.

**Spec:** The README and phase_00 through phase_11-12 markdown files in this directory's parent. These supplied documents are the implementation specification. Differences required for exact reference compatibility are logged in decisions.md.

## Tasks, interfaces, and verification

- [x] Phase 0–2: conventions.py, geometry.py and stages.py. `Geometry(n,delta,canvas_shape,frame_shape,margin)` is a pytree with static shapes. `exit_waves(probe,obj,a,geom)` returns `(shots,modes,rows,columns)`. Compare values and gradients against the untouched reference, fixed seed golden archive, integer/fractional positions, float32 dtype and crop validity.
- [x] Phase 3: operators.py. Implement extraction/scatter, exact shifted-object map/adjoint, object/probe linear maps and Hermitian adjoints, and nonnegative normal diagonals. Dot tests and independent jax.linear_transpose checks precede projector implementation.
- [x] Phase 4–5: projections.py. Exact shared multimode Fourier rescaling with mask and deterministic zero-amplitude fallback. Damped object CG, exact probe division, nonnegative fixed/global/per-shot amplitude solves, warm-started alternating overlap projection. Verify constraints, fixed points, KKT, objective descent, zero-weight modes and traced gates.
- [x] Phase 6–7: gauge.py and init.py. Product-preserving normalization, frozen collapsed modes and seeded QR initialization. Gate housekeeping so frozen probe/weight blocks remain bit-identical. Initialize measured masked counts exactly.
- [x] Phase 8–9: dm.py and metrics.py. Beta=1 and general-beta difference maps, traced schedules, scan blocks with donation, numerical failure dump, Fourier/model/update/weight metrics. Compare with an independent NumPy reference and test trace counts across stage boundaries.
- [x] Phase 10–11: tests/synthetic.py, scripts/run_validation_ladder.py and handoff.py. Run A–F plus frozen-weight control, gauge-aware object and coherence metrics and diagonal-amplitude-to-full-matrix handoff. Freeze positions during AD, use the untouched forward/loss/optimizer APIs, and compare warm/naive starts.
- [x] NanoMAX demonstration: add src/pipeline_nanomax_dm_ad.py. Load the validated 169-shot 512-pixel NanoMAX data once. DM optimizes object, two modes and shot weights; AD optimizes object, modes and full modal matrices while positions remain fixed. Save initialization/DM/AD snapshots, curves, matrices, measured errors and configuration for correct and +30% defocus, with AD-only controls.
- [x] Phase 12 and completion: measure timing/memory on three sizes, compare full/chunked operators, document single-device scope and unsupported optional paths explicitly, write per-phase reports and final implementation progress/results. Re-run all new tests and the existing data-loading/pipeline tests. Verify old file hashes.

Tests live in `src/ptycho_jax/DM/tests/` and run with `PYTHONPATH=src .venv/bin/python -m pytest --confcutdir=src/ptycho_jax/DM/tests src/ptycho_jax/DM/tests -q`. Algebra tests enable x64; production inputs explicitly use float32/complex64.

Core implementation and requested experiments are complete. Optional canonicalization heuristics, translation alignment and operation-level profiling are explicitly deferred in the progress and phase reports.
