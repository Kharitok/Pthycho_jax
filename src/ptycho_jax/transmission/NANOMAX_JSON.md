# NanoMAX JSON settings reference

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
  --config configs/nanomax_scheduled.json
```

The [example JSON](../../../configs/nanomax_scheduled.json) contains every
supported setting and runs 400 updates with five probe modes, scheduled sample
TV, scheduled probe support and full diagnostics. Edit `datapath` for your data.
The general [pipeline guide](NANOMAX_V2.md) explains the reconstruction outputs.

## Input rules and precedence

The file is one flat JSON object. Keys use the Python names below, with
underscores. All fields are optional: omitted fields keep the documented defaults.
Use JSON `true`/`false`, numbers, arrays and `null`; comments, duplicate keys,
unknown keys, NaN/Infinity and numeric/boolean strings are rejected.

Supplied relative `datapath` and `output_dir` values are resolved relative to
the JSON file's directory; `~` is expanded. An omitted `output_dir` keeps the
CLI default `results/nanomax` relative to the current working directory.
Explicit CLI arguments override corresponding JSON values and relative CLI
paths use the current working directory:

```bash
PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
  --config configs/nanomax_scheduled.json --tag another_run --device cpu
```

Types must be valid in the JSON itself; value constraints and schedule bounds
are checked after merging overrides. A run creates a fresh timestamp/tag folder.
The exact JSON bytes, SHA-256, original path, invocation arguments and effective
settings are preserved there. A saved output `configuration.json` also contains
generated metadata and is **not** an input file; edit the saved `input_settings.json`
copy instead. Paths in a copied input remain relative to its new location, so
adjust them or make them absolute before moving the file.

## Run, data and geometry

Defaults in this reference are script defaults, which can differ from the
enabled settings in the example. All physical lengths use metres unless noted.

| JSON key | Type and default | Meaning |
| --- | --- | --- |
| `datapath` | string; local scan `.../NanoMax/Test_reconstruction/download_data/09_kev/raw/0001_setup/000082.h5` | NanoMAX raw HDF5 scan. The example contains the full local path; supply your own path on another machine. |
| `detector_name` | string; `"eiger4m"` | Detector group under `/entry/measurement/<name>/frames`. Use `"eiger1m"` for EH2 scan 000005. HDF5 external links are followed to the detector file automatically; motors and energy still come from `datapath`. |
| `shot_specific_masks` | boolean; `false` | Estimate the existing high-value exclusion mask independently for each full raw frame, then crop/orient it with that frame and use it in the loss. The maintained examples set this to `true`. `false` reuses the first-frame mask. See details below. |
| `output_dir` | string; `"results/nanomax"` | Parent directory for a new `scan_<scan>_<UTC timestamp>_<tag>` folder. Existing run folders are rejected. |
| `tag` | string or `null`; five random ASCII letters | Run label shared by every artifact filename. Spaces become underscores in paths. Omit or use `null` for a new random label. |
| `device` | `"gpu"`, `"cpu"` or `null`; `null` | JAX execution backend. `null` selects JAX's default available device. |
| `modes` | array of distinct positive integers; `[1, 5]` | Separate reconstructions to perform after loading data once. `[5]` runs one five-mode reconstruction. |
| `n_steps` | positive integer; `400` | Number of full-pass AD updates per reconstruction. Each uses every retained frame and one Adam update. |
| `steps` | positive integer; alias for `n_steps` | Alternative spelling matching CLI `--steps`. Do not supply both names. |
| `batch_size` | positive integer; `13` | Frames per gradient batch. A partial final batch is included and correctly weighted. This does not change schedule indexing. |
| `seed` | nonnegative integer; `0` | Seed for initialization noise and frame shuffling. Random default tags are independent of this seed. |
| `detector_distance_m` | positive number; `7.18` | Sample-to-detector distance used in geometry and probe initialization. |
| `detector_pixel_size_m` | positive number; `0.000075` | Physical detector pixel pitch. |
| `defocus_m` | finite signed number; `0.0015` | Supplied focus-to-sample distance for probe initialization; positive means sample downstream of focus. Also sets the default diagnostics range. |
| `crop_shape` | two positive integers; `[512, 512]` | Detector crop size `[rows, columns]`. Production examples use square crops. |
| `crop_center` | two numbers or `null`; `null` | Crop center `[row, column]` in raw detector pixels, before row flipping. `null` estimates it from the selected data. |
| `scan_coords_unit` | positive number; `0.000001` | Metres per raw motor coordinate unit; the NanoMAX example motors are in micrometres. |
| `energy_unit` | positive number; `1.0` | Factor converting raw energy metadata to eV. Wavelength is calculated from that energy. |
| `motor_x_sign` | integer `-1` or `1`; `1` | Sign applied to the x beam coordinate derived from raw sx. |
| `motor_y_sign` | integer `-1` or `1`; `-1` | Sign applied to the y beam coordinate derived from raw sy. Array coordinates use `(motor_y_sign*sy, motor_x_sign*sx)`. |
| `flip_detector_rows` | boolean; `true` | Reverse detector rows before the pipeline's FFT-order conversion. |
| `scan_coord_limits` | object or `null`; `null` | Inclusive limits in **original raw motor units**, before sign/orientation changes. Keys `"0"` and `"1"` mean sx and sy. Example: `{"0": [-1.0, 1.0], "1": [-2.0, 2.0]}`. Each pair must be finite and ordered; either axis may be omitted. |
| `intensity_limits` | two numbers or `null`; `null` | Inclusive `[minimum, maximum]` cleaned/masked/cropped photon-total limits per frame. Pairs must be finite and ordered. `null` disables this filter. |
| `max_positions` | positive integer or `null`; `null` | Retain at most this many frames in reconstruction order, after filtering. `null` uses all retained frames. Useful for a quick smoke test. |

### Shot-specific detector masks

Set `"shot_specific_masks": true` to give every exposure its own validity mask.
The criterion is unchanged: a pixel is valid when its **raw full-frame** value
is finite, nonnegative and strictly below that frame's largest finite value.
All pixels tied at that maximum are excluded. This is a heuristic for the
high-valued invalid/saturated-pixel sentinel: if a frame's maximum is genuine
signal, those brightest pixels are excluded too. There is no new intensity
threshold. A frame with no valid crop pixels raises an error.

Masks remain aligned through coordinate/intensity filtering, repeat sorting,
position limits, cropping, row flipping, FFT shifting and shuffled batches.
Each shot's data loss is averaged over its own valid pixels; totals and initial
probe-weight calibration use exactly those pixels. Automatic centering and
probe-initialization means average each pixel over only the exposures where it
is valid. Masks are fixed data, not optimized parameters.

The saved `shot_masks` is a boolean `(frames, rows, columns)` array in
reconstruction/FFT order, aligned with `frame_indices`; `True` means valid.
`detector_mask` is its **2D union**, used for shared spatial support/display,
not as a replacement for the per-shot loss masks. The preparation image shows
the fraction of retained shots observing each detector pixel. Metrics record
mean/min/max shot validity fractions. This adds one boolean crop per frame.

Set `"shot_specific_masks": false`, or use `--no-shot-specific-masks`, to keep
the legacy shared first-frame behavior. In this mode saved `shot_masks` has an
empty leading dimension. CLI `--shot-specific-masks` enables the option and
overrides JSON. Python callers may additionally supply static `mask` and
`precise_mask` arrays: with shot masks enabled these intersect every shot's
estimated mask; with the option disabled, explicit `mask` replaces the heuristic.

## Initialization and optimization

| JSON key | Type and default | Meaning |
| --- | --- | --- |
| `shift_margin` | integer; `10` | Object-selector margin in pixels around the extracted probe-sized patch. Must exceed `max_correction_magnitude`. |
| `max_correction_magnitude` | number; `5.0` | Pixel bound for the selector's `tanh` Fourier-shift parameterization. Require `1 < value < shift_margin`. |
| `object_padding` | positive integer; `32` | Additional padding around the sample canvas in pixels. |
| `relative_probe_threshold` | number in `[0,1)`; `0.005` | Retain mean detector intensities strictly above this fraction of their maximum when initializing the probe. |
| `sample_learning_rate` | nonnegative number; `0.03` | Adam learning rate for the complex sample. Zero disables its gradient updates. |
| `probe_learning_rate` | nonnegative number; `0.3` | Adam learning rate for global probe modes. Zero disables their gradient updates; enabled support projections can still modify them. |
| `weights_learning_rate` | nonnegative number; `0.003` | Adam learning rate for shot-dependent complex modal-weight matrices. Zero disables their gradient updates. |
| `positions_learning_rate` | nonnegative number; `0.003` | Adam learning rate for scan-correction latents. Zero keeps the initial fractional geometry fixed. |

Zero learning rates do not disable scheduled projections. Use `probe_support`
or `projection_schedule` to control support independently of gradient updates.

## TV regularization

| JSON key | Type and default | Meaning |
| --- | --- | --- |
| `sample_tv_weight` | nonnegative number; `0.0` | Weight on complex sample TV; zero disables that term. |
| `probe_tv_weight` | nonnegative number; `0.0` | Weight on complex probe-mode TV; zero disables that term. |
| `tv_p` | positive number; `2.0` | Spatial-difference power in the existing generalized TV function. With q=1, p=2 gives smoothed isotropic TV. |
| `tv_q` | positive number; `1.0` | Power applied to the generalized gradient magnitude before averaging. |
| `tv_epsilon` | positive number; `0.000001` | Smooth absolute-value epsilon: each complex difference uses `sqrt(abs(diff)^2 + epsilon^2)`. |
| `tv_normalize` | boolean; `true` | Divide each spatial array by its RMS amplitude before TV, separately for each probe mode. This prevents reducing TV through compensated amplitude rescaling. `false` selects raw-amplitude TV; it does **not** disable TV. |
| `regularization_schedule` | array of strings or `null`; `null` | Updates at which configured TV terms, and Python custom regularizers if used, participate in gradients. See schedule syntax below. |

TV acts on complex differences, jointly affecting amplitude and phase. Its
per-array value is the mean of `(smooth_abs_dx^p + smooth_abs_dy^p + 1e-20)^(q/p)`.
Penalties are counted once per full pass, not once per batch. Arbitrary modal-basis
dependence remains even with RMS normalization.

## Probe-support projection

| JSON key | Type and default | Meaning |
| --- | --- | --- |
| `probe_support` | boolean; `false` | Enable a shared real amplitude support on every probe mode after selected Adam updates. |
| `probe_support_radius_px` | positive number or `null`; `null` | Flat support radius in sample-plane array pixels. `null` means `0.4 * min(crop_shape)`. |
| `probe_support_decay_px` | positive number or `null`; `null` | Cosine-taper width in pixels beyond the flat radius. `null` means `0.1 * min(crop_shape)`. The support is exactly zero at radius+decay and beyond. |
| `probe_support_center_px` | two numbers or `null`; `null` | Support center `[row, column]` in sample-plane probe-array pixels. `null` means `[(rows-1)/2, (columns-1)/2]`. |
| `projection_schedule` | array of strings or `null`; `null` | Updates after which the enabled support and any Python custom projection are applied. Independent of the TV schedule. |

Repeated support application repeatedly attenuates the tapered region. Null
radius/decay values select automatic dimensions; they do not disable projection.

## Additive detector background

The background is optional and disabled by default. With it enabled,
`I_pred[j] = I_signal[j] + frame_counts[j] * probability`, after intensity
binning and before sensor response. The shared probability image is
nonnegative, zero on invalid pixels, and sums to one on valid detector pixels.
Its coarse logits are interpolated in centered detector coordinates, then
converted to the model's unshifted FFT order. This restricts fine structure
that could otherwise mimic useful diffraction features.

`frame_counts[j] = measured_total[j] * background_max_fraction * sigmoid(level_logit[j])`.
The bound is structural and remains active even when regularizers/projections
are scheduled off. Initial probe weights are calibrated against measured
photons minus the initial background, avoiding double counting.

| JSON key | Type and default | Meaning |
| --- | --- | --- |
| `background` | boolean; `false` | Optimize a single shared additive detector background with a level for each frame. `false` retains the original signal-only model. |
| `background_grid_shape` | two integers >=2; `[16,16]` | Shared log-density grid. Each dimension must fit within the detector crop when enabled. Smaller grids limit background complexity more strongly. |
| `background_initial_fraction` | positive number; `0.0001` | Initial fraction of each frame's measured valid-pixel photons assigned to background. Must be below the maximum. A small positive value gives usable gradients. |
| `background_max_fraction` | number in `(initial_fraction,1)`; `0.05` | Maximum allowed background fraction of each frame's measured valid-pixel photon count. This is not a per-pixel or predicted-intensity bound. |
| `background_learning_rate` | nonnegative number; `0.03` | Adam learning rate for shared shape logits. Zero freezes this shape's gradient updates. |
| `background_scale_learning_rate` | nonnegative number; `0.03` | Adam learning rate for per-frame level logits. Zero freezes these levels' gradient updates. |
| `background_start_step` | nonnegative integer; `100` | First full-pass update allowed to change background parameters. The small initialized background is modeled before this step. If >=`n_steps`, it stays at initialization. |
| `background_l1_weight` | nonnegative number; `0.01` | Penalizes mean background counts per valid pixel, preferring the smallest background supported by data. |
| `background_smoothness_weight` | nonnegative number; `1.0` | Penalizes squared spatial gradients of the unit-mean shared shape. Edges touching invalid detector pixels are excluded. |
| `background_scale_weight` | nonnegative number; `0.1` | Penalizes variance of log per-frame background photon counts, discouraging excessive scale jitter while permitting genuine changes. |

The three penalties operate on physical normalized quantities, not arbitrary
raw-logit scales. They follow `regularization_schedule` together with TV and
custom penalties. For background fitting, keep this schedule active unless
you deliberately want unregularized intervals. Setting a weight to zero turns
off only that penalty; a coarse shape, nonnegativity and the photon cap still
apply. Set `"background": false` or CLI `--no-background` to remove the component
entirely. Freezing both background learning rates retains its initial counts;
it does not remove the component.

Saved physical arrays are `background_map` (mean counts/pixel),
`background_scales` (mean one), `background_frame_counts` and
`background_fractions`; `scale[j] * background_map` is the fitted additive
image for frame j. The optimizer leaves are `background_logits` (coarse shared
shape) and `background_scale_logits` (per-frame levels).

With shot-specific masks, the background remains **one common spatial image**;
it is not renormalized separately for each shot. The full union-support photon
count is capped as above, so background on the observed subset is also bounded.
`background_frame_counts` records full-map counts;
`background_observed_frame_counts` records counts on each shot's valid pixels.
`background_fractions` and photon-fraction audits use observed counts.
`background_mean_observed_map` averages masked background contributions across
shots and is used in mean signal/background comparisons. L1 uses observed
background counts per observed pixel; spatial smoothness uses union support.
The `fraction_frames_near_budget` audit checks the fitted full-map level against
its cap, even when some of that map is unobserved in a particular shot.

The report and background figures separate signal/background, show frame
scales in acquisition order, and audit mean/max background photon fractions,
bright-pixel leakage, budget saturation and dark-pixel dominance. Background
may exceed the signal in dark pixels without dominating useful photons.
Photon caps and smoothness reduce ambiguity; preserved object features and
comparison with a signal-only control remain important checks.

Example with TV off and background penalties active at all updates:

```json
{
  "background": true,
  "background_grid_shape": [16, 16],
  "background_max_fraction": 0.05,
  "background_start_step": 100,
  "regularization_schedule": ["all"],
  "sample_tv_weight": 0.0,
  "probe_tv_weight": 0.0
}
```

## Diagnostics

The background example can be run with
`--config configs/nanomax_eh2_background.json`. It was executed with 1000
updates/five modes on EH2 scan 000005. Fitted background was **0.0007384%** of
measured photons on average (**0.0007922%** maximum frame), and **0.0002980%**
of predicted photons in the bright detector region. Phase correlation to the
stored signal-only control was **0.999981**, with phase-contrast slope **1.00092**.
This confirms no useful-signal dominance in that run; it does not establish a
uniquely calibrated physical background without independent background data.
Outputs are under
`results/nanomax_eh2/scan_000005_20261123T232649Z_eh2_background_ad/`;
start with its prefixed `background_validation.md` or `results.md`.

| JSON key | Type and default | Meaning |
| --- | --- | --- |
| `diagnostics` | boolean; `true` | Run final SVD/coherent-mode analysis, beam propagation, focus estimation, scan/drift and beam-evolution diagnostics. `false` skips analysis while retaining reconstruction outputs. |
| `half_range_m` | positive number or `null`; `null` | Propagate over `[-half_range_m, +half_range_m]` about the sample. `null` means `2*abs(defocus_m)`, ±3 mm by default. For zero defocus supply a positive value if diagnostics are enabled. |
| `z_samples` | odd integer >=5; `201` | Number of propagation planes, including the sample plane at z=0. |
| `padding_factor` | positive integer; `3` | Transverse zero-padding factor during propagation, reducing periodic wraparound. |

Orthogonalization transforms analysis copies and compensates modal weights,
preserving fitted fields. Focus minimizes centered second-moment trace;
positive z is downstream of the sample. Drift slopes restore acquisition order
and are per shot, not elapsed-time velocities. These are reconstruction-derived
diagnostics rather than independent calibrations.

## How to change the schedules

**Indices are zero-based: step 0 is the first full-pass update. STOP is exclusive.**
Each schedule is a JSON array of strings. Multiple entries form a union; overlaps
apply the operation only once. Blank arrays `[]` are invalid. The entries are:

| JSON schedule value | Behavior for 400 updates |
| --- | --- |
| `null` or `["all"]`, or omit the field | Every update, if that operation is enabled |
| `["none"]` | No penalty/projection applications; background parameter learning is controlled separately |
| `["50:350"]` | Updates 50 through 349: 300 applications |
| `["0:400:10"]` | Updates 0,10,…,390: every tenth update |
| `["100:400:5", "399"]` | Updates 100,105,…,395 plus the final update: 61 applications |
| `["0:100", "300:"]` | First 100 and last 100 updates |
| `[":100"]` | First 100 updates |
| `["399"]` | Final update only |

`START:STOP:EVERY` anchors spacing at START. EVERY must be positive. Ranges must
lie within `[0, n_steps)`; if you shorten a reconstruction, update its schedules
too. `all`/`none` must be alone. TV is added during the selected update's gradient
calculation. Projection occurs immediately after the selected Adam update.
All batches in a full pass use the same setting; Adam and shuffle state continue
through switches.

For example, apply TV early and support only during the final refinement:

```json
{
  "n_steps": 400,
  "sample_tv_weight": 1.0,
  "regularization_schedule": ["0:200"],
  "probe_support": true,
  "probe_support_radius_px": 140.0,
  "probe_support_decay_px": 40.0,
  "projection_schedule": ["200:400"]
}
```

Turning an operation off does not undo earlier updates. Probe values outside
support can regrow during later unprojected steps; include the final index if
you want the saved probe to satisfy the support exactly. Objective-history jumps
can reflect TV switching. The final objective uses the last update's TV setting;
`final_configured_regularization` records weighted TV before schedule gating.
The saved activity arrays/counts and schedule figure show what actually ran.

## How to switch TV and projections off

Disable both built-in TV penalties and the support:

```json
{
  "sample_tv_weight": 0.0,
  "probe_tv_weight": 0.0,
  "probe_support": false
}
```

Alternatively, keep all strengths/dimensions and disable their schedules:

```json
{
  "regularization_schedule": ["none"],
  "projection_schedule": ["none"]
}
```

To disable only sample TV, set its weight to zero and leave probe TV unchanged.
To re-enable TV, provide a positive weight and an active schedule. To re-enable
support, set `probe_support` to true and provide an active projection schedule.
A schedule alone does not enable a zero-weight penalty or disabled support.

From the command line, override a JSON configuration without editing it:

```bash
PYTHONPATH=src .venv/bin/python src/pipeline_nanomax_upd_v2.py \
  --config configs/nanomax_scheduled.json \
  --regularization-schedule none --no-probe-support
```

Use `"diagnostics": false` or CLI `--no-diagnostics` to skip final diagnostics;
that setting does not disable TV or projections. `started_utc`, `run_id`, source
hashes and other generated report fields are output metadata, not JSON inputs.
The run start timestamp is always generated afresh.

If additive background is enabled, `regularization_schedule: ["none"]` turns
off its penalties while its parameter learning continues. Set `"background": false`
to remove background entirely; see its dedicated parameter section above.

## Executed JSON example

The command at the top of this document was run with the supplied JSON on
2026-11-23: 400 updates, five modes, 169 frames, 300 TV-active updates and
61 projections, followed by all diagnostics. Data MSE fell from **193.70151**
to **3.51730**; joint RMS focus was **−1.43949 mm from the sample**.

Outputs: `results/nanomax/scan_000082_20261123T110102Z_json_settings_test/`.
Open its `20261123T110102Z_json_settings_test_results.md`. The prefixed
`input_settings.json` is the exact input copy; `verification.json` records
the setting-by-setting comparison, input/source hashes, actual schedule masks,
numeric-array checks and report-link checks.
