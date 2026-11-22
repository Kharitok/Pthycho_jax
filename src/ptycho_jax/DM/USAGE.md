# Running difference-map reconstruction

Run from the repository root with the existing project environment. The source
AD model and previous NanoMAX pipeline versions are unchanged.

## Two defocus guesses, DM followed by AD

```bash
.venv/bin/python src/pipeline_nanomax_dm_ad.py \
  --output-dir results/my_dm_ad_run \
  --dm-iterations 200 --ad-steps 400 --dm-inner 3 --cg-iterations 5 \
  --batch-size 13 --defocus-factors 1.0 1.3 --device gpu
```

This loads the known NanoMAX scan, uses two probe modes and shot-specific
nonnegative amplitude weights in DM, and compares 1.50 mm and 1.95 mm probe
initialization. Both stages keep the scan positions fixed. The output directory
must be new. Change input data and geometry with `--datapath`, `--crop-shape`,
`--crop-center`, `--defocus-m`, `--detector-distance-m` and
`--detector-pixel-size-m`. The provided defaults preserve the validated NanoMAX
energy and coordinate conventions.

The default AD continuation uses conservative learning rates (sample/probe/
weights: 0.003 / 0.03 / 0.0003). To repeat the additional control with the earlier
AD pipeline's original rates, without recomputing DM:

```bash
PYTHONPATH=src .venv/bin/python src/ptycho_jax/DM/scripts/compare_ad_rates.py \
  results/my_dm_ad_run --epochs 400
```

Start with `results.md`, then `original_ad_rates_comparison.md`. Each defocus
case contains initial, DM, AD and AD-only parameter archives and separate object,
probe, shot-weight, convergence and detector-comparison figures. The optional
rate control writes `ad_original_rates_warm/` and `ad_original_rates_naive/`.

## Standalone DM, without AD

```bash
PYTHONPATH=src .venv/bin/python src/ptycho_jax/DM/scripts/run_nanomax_dm.py \
  --output-dir results/my_dm_only --iterations 200 --crop 512 \
  --probe-defocus-m 0.0015
```

`parameters.npz` contains a reference-compatible object, probe modes and diagonal
shot-amplitude matrices. `history.npz` contains Fourier error, overlap-model
MSE/R, update norm, weight statistics and inactive-mode counts. Raw exit-wave
iterates are not archived by these scripts; the saved overlap factors are what
AD must consume. Restarting the raw DM trajectory would also require its psi.

## Library API

```python
import pipeline_nanomax_upd as reference
from ptycho_jax.DM.config import DMConfig
from ptycho_jax.DM.nanomax import make_start
from ptycho_jax.DM.dm import run_dm
from ptycho_jax.DM.handoff import to_ad_parameters, refine_ad

cfg = reference.ReconstructionConfig()
loaded = reference.load_reconstruction_data(cfg)
state, data, geometry, origin = make_start(loaded, cfg, guess_defocus=1.5e-3)
state, dm_history = run_dm(state, data, geometry, DMConfig(), 200)
params, fixed = to_ad_parameters(state, geometry)
params, ad_history, statistics = refine_ad(params, fixed, geometry, data)
```

`run_dm` donates state buffers by default: do not access the input state after
calling it. Initial guesses used by independent controls must be copied first;
`to_ad_parameters` performs that copy. All large data and geometry arrays are
explicit compiled-function arguments.

## Validation

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  --confcutdir=src/ptycho_jax/DM/tests src/ptycho_jax/DM/tests -q

PYTHONPATH=src .venv/bin/python src/ptycho_jax/DM/scripts/run_validation_ladder.py \
  --output results/my_dm_ladder --iterations 200 --ad-steps 50

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 PYTHONPATH=src \
  .venv/bin/python src/ptycho_jax/DM/scripts/check_sharding.py
```

Algebra tests use x64. Production uses float32/complex64 and requests highest
float32 contraction precision. Optional shot-axis sharding is provided by
`sharding.shard_positions`; shot count must divide device count. The 169-shot
NanoMAX experiment intentionally uses one GPU.
