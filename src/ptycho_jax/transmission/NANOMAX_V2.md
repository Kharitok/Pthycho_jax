# NanoMAX AD with scheduled constraints and diagnostics

Run `src/pipeline_nanomax_upd_v2.py` from the repository root. It shares the
validated loader, geometry, initialization, optimizer and plots with
`src/pipeline_nanomax_upd.py`. The older entry point keeps its behavior.

Settings can also come entirely from JSON:

```bash
PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
  --config configs/nanomax_scheduled.json
```

See the [JSON parameter reference](NANOMAX_JSON.md) for every field, schedule
examples, disabling constraints, path rules and CLI overrides.

```bash
# Five-mode, 400-update AD with scheduled TV/support and full diagnostics.
PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
  --modes 5 --steps 400 --device gpu --tag scheduled_diagnostics_test \
  --sample-tv-weight 1.0 --regularization-schedule 50:350 \
  --probe-support --probe-support-radius-px 140 --probe-support-decay-px 40 \
  --projection-schedule 100:400:5 399 \
  --half-range-m 0.003 --z-samples 201 --padding-factor 3

# TV at every update, without support; skip diagnostics if desired.
PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
  --modes 5 --steps 400 --sample-tv-weight 1.0 --no-diagnostics --tag tv_only
```

`--output-dir` is a **parent** directory (default `results/nanomax`). Each
invocation makes `scan_<scan>_<YYYYMMDDTHHMMSSZ>_<tag>/`. The UTC timestamp is
captured once, before loading, and reused in every artifact filename, including
source snapshots. Spaces in tags become underscores; the original tag is
preserved in JSON and NPZ metadata. Without `--tag`, five random ASCII letters
are used. Existing run folders are rejected instead of overwritten.

An AD step is one complete pass over all frames followed by one Adam update.
The default learning rates remain sample/probe/weights/positions
`0.03 / 0.3 / 0.003 / 0.003`. Position refinement remains enabled.

`--regularization-schedule` and `--projection-schedule` independently select
updates. **Step 0 is the first update; STOP is exclusive.** Multiple entries
form a union; intervals outside the reconstruction budget are rejected.

| Schedule | Active updates |
| --- | --- |
| omitted or `all` | Every update, if the corresponding constraint is enabled |
| `none` | No updates |
| `50:350` | 50 through 349: 300 updates |
| `100:400:5 399` | 100,105,…,395 plus 399: 61 updates |
| `0:100 300:` | First 100 updates and updates from 300 to the end |

A schedule does not enable zero-weight TV or an absent `--probe-support`.
Regularization participates in the scheduled update's gradient; projection
acts immediately after that update. Adam state and the frame-shuffling stream
continue across switches. Turning a constraint off affects subsequent updates;
it does not undo earlier effects. A support boundary can regrow on later
unprojected steps; include the final index (e.g. `399`) to enforce it at the end.

The objective is masked amplitude MSE plus optional weighted complex TV of
the sample (`--sample-tv-weight`) and probe modes (`--probe-tv-weight`). Zero
weights disable these terms. `--tv-p`, `--tv-q` and `--tv-epsilon` default to
`2`, `1` and `1e-6`. The functions live in `transmission/loss_and_reg.py`.
Penalties are attached to the mean batch loss so full-pass averaging counts
them once, including a partial final batch.

By default, each spatial array is divided by its RMS amplitude before TV;
each probe mode is normalized separately. This prevents a penalty reduction
through compensated sample/probe/modal-weight amplitude rescaling. The
normalization is differentiable and safe at zero. `--no-tv-normalize` selects
the original raw-amplitude TV convention. Arbitrary modal-basis dependence
remains; these penalties do not by themselves identify a unique physical basis.

`--probe-support` multiplies every probe mode by the same real support **after
scheduled Adam updates**. The support equals one up to the chosen radius, follows a
cosine taper over the decay width, and is exactly zero beyond their sum.
Coordinates are sample-plane array pixels. The default center is the geometric
probe center; `--probe-support-center-px ROW COL` overrides it. Default radius
and decay are `0.4` and `0.1` times the smaller crop dimension. Repeated
application repeatedly attenuates the tapered region. The support is saved as
an array and a figure, with actual dimensions and initial retained probe power
recorded in metrics. Both regularization and support can be enabled together.

Python callers can use `ReconstructionConfig` and `run_pipeline(config, [5])`.
The config owns `run_id`, `run_dir` and `artifact(name, folder=None)` so all
products share one identity. `regularizer_fn(params, fixed) -> scalar` adds a
custom penalty. `projection_fn(params, fixed) -> (params, fixed)` adds a
custom post-update constraint; the configured probe support is applied last.
These callbacks should be pure JAX-compatible functions.
They follow the corresponding schedule and must preserve dictionary structure.
For Python use, schedules are tuples such as `regularization_schedule=("50:350",)`.

**End-of-run diagnostics are enabled by default.** They save SVD-orthogonalized
and ensemble coherent modes with compensated weights (preserving every fitted
shot field), coherent-mode power fractions, HSV beam images, Fresnel propagation,
joint/x/y focus estimates, scan corrections, drift trends, modal populations and
beam-centroid/width evolution. Orthogonalization operates on analysis copies;
the optimized parameters remain in the reconstruction archive.

`--half-range-m` is the propagation half-range **about the sample**; its default
is `2*abs(defocus)` (±3 mm for this scan). `--z-samples` defaults to 201 and must
be odd and at least 5; `--padding-factor` defaults to 3. Positive propagation z
is downstream. Focus minimizes the beam's centered second-moment trace. Drift
plots restore acquisition frame order; slopes are nm per shot, not temporal
velocities. `--no-diagnostics` skips this stage.

Each run saves a linked Markdown report, configuration, summary, preparation
and comparison plots, source snapshots with SHA-256 hashes, and per-mode
sample/probe/diffraction/objective/schedule figures plus reconstruction NPZ and
metrics. Start with the prefixed root `results.md`; it links each mode's full
`diagnostics/` report, separate images, numerical arrays and summary.
`loss_history` and `objective_history` contain the **total objective before
each update**. Initial/final data MSE and regularization are saved separately;
`initial_loss` and `final_loss` in metrics always mean data MSE alone. The last
plot marker evaluates final parameters using the last update's TV setting.
`final_configured_regularization` retains the weighted TV before schedule gating,
even when TV is off at the last update. Switching TV can cause objective jumps.
NPZ activity masks (`regularization_active_history`, `projection_active_history`)
and metric counts record the schedule actually used.

The combined example was executed on 2026-11-23. Its outputs are in
`results/nanomax/scan_000082_20261123T103635Z_scheduled_diagnostics_test/`:
data MSE **193.70151 → 3.51730**, joint RMS focus **−1.43949 mm from the sample**.
Start with that folder's `20261123T103635Z_scheduled_diagnostics_test_results.md`.

Run the versioned regression checks without the legacy root test conftest:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python -m pytest \
  --confcutdir=src/ptycho_jax/transmission/tests \
  src/ptycho_jax/transmission/tests -q --tb=short
```
