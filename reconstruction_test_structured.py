"""
Interactive-style end-to-end reconstruction test for transmission ptychography.

Cells are organized as:
1) test parameters + forward model construction
2) forward simulation (synthetic data)
3) reconstruction config
4) reconstruction setup (loss/regularization/optimizer/projections)
5) reconstruction run
6) result display
"""

# %%
# Imports and shared helpers
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from skimage import data

import projectors as pj
from loss_and_reg import (
    combine_regularizers,
    create_regularizer,
    create_tv_reg,
    gauss_loss,
    join_loss_and_forward_batched,
    join_loss_and_reg,
)
from optim_loop import run_optimization_loop
from optimizers import prepare_optimizer
from transmission_ptycho_models import construct_point_model_ptycho_transmission


@dataclass(frozen=True)
class ForwardConfig:
    obj_size: int = 128
    probe_size: int = 64
    n_positions: int = 100
    n_modes: int = 3
    probe_type: str = "static"
    sample_type: str = "complex"
    sample_selector_type: str = "correcting"
    propagator_type: str = "Fourrier"
    shift_margin: int = 10
    max_correction_magnitude: float = 5.0
    measurement_noise_std: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class ReconstructionConfig:
    reconstruction_type: str = "accumulate_full_pass"
    n_steps: int = 300
    batch_size: int = 50
    shuffle_each_epoch: bool = True
    use_multigpu: bool = False
    seed: int = 1

    # Initialization noise
    position_noise_px: int = 2
    init_scan_mistake_std: float = 0.3
    init_probe_noise_std: float = 20.0

    # Regularization
    tv_sample_weight: float = 0.0
    tv_probe_modes_weight: float = 0.0

    # Projection knobs
    shift_rounding_step: float = 3.0
    max_allowed_whole_shift: float = 10.0


def _build_probe_modes(cfg: ForwardConfig) -> jnp.ndarray:
    gravel = data.gravel().astype(np.float32)
    modes = []
    for mode_idx in range(cfg.n_modes):
        y0 = cfg.probe_size * mode_idx
        y1 = cfg.probe_size * (mode_idx + 1)
        x0 = cfg.probe_size * mode_idx
        x1 = cfg.probe_size * (mode_idx + 1)
        modes.append(gravel[y0:y1, x0:x1])

    modes = np.stack(modes, axis=0)
    probe_mask = np.zeros((cfg.probe_size, cfg.probe_size), dtype=np.float32)
    probe_mask[
        cfg.probe_size // 4 : 3 * cfg.probe_size // 4,
        cfg.probe_size // 4 : 3 * cfg.probe_size // 4,
    ] = 1.0

    phase = np.swapaxes(modes / (np.max(np.abs(modes)) + 1e-12), -1, -2)
    modes = np.abs(modes) * np.exp(1j * 2 * np.pi * phase)
    modes = modes * probe_mask[None, :, :]

    q, _ = np.linalg.qr(modes.reshape(cfg.n_modes, -1).T, mode="reduced")
    modes = q.T.reshape(cfg.n_modes, cfg.probe_size, cfg.probe_size)
    return jnp.asarray(modes)


# %%
# 1) Test parameters + forward model construction
forward_cfg = ForwardConfig(
    obj_size=128,
    probe_size=64,
    n_positions=100,
    n_modes=3,
    sample_selector_type="correcting",
    shift_margin=10,
    measurement_noise_std=0.0,
    seed=0,
)

rng_forward = np.random.default_rng(forward_cfg.seed)
image = data.camera().astype(np.float32)
image = zoom(
    image,
    (forward_cfg.obj_size / image.shape[0], forward_cfg.obj_size / image.shape[1]),
    order=3,
)

true_probe_modes = _build_probe_modes(forward_cfg)
true_sample = (0.5 + np.abs(image) / np.abs(image).max()) * np.exp(
    1j * 2 * np.pi * (image.T / image.max())
)
true_sample = true_sample / (np.abs(true_sample).max() + 1e-12)

true_scan_positions = rng_forward.integers(
    0,
    forward_cfg.obj_size - (5 + forward_cfg.probe_size),
    size=(forward_cfg.n_positions, 2),
    endpoint=False,
)

true_diff_params = {
    "sample": jnp.asarray(true_sample),
    "probe_modes": true_probe_modes,
    "modal_weights": jnp.ones(
        (forward_cfg.n_positions, forward_cfg.n_modes, forward_cfg.n_modes),
        dtype=jnp.complex64,
    ),
    "scan_mistakes": jnp.zeros((forward_cfg.n_positions, 2), dtype=jnp.float32),
}
true_non_diff_params = {
    "sample_positions": jnp.asarray(true_scan_positions, dtype=jnp.int32),
}

model_parameters = {
    "probe_type": forward_cfg.probe_type,
    "sample_type": forward_cfg.sample_type,
    "binning_factor": (1, 1),
    "shift_margin": forward_cfg.shift_margin,
    "sample_selector_type": forward_cfg.sample_selector_type,
    "propagator_type": forward_cfg.propagator_type,
    "probe_shape": (forward_cfg.probe_size, forward_cfg.probe_size),
    "max_correction_magnitude": forward_cfg.max_correction_magnitude,
}
models = construct_point_model_ptycho_transmission(model_parameters=model_parameters)
point_model = models["point_model"]

print("Forward model built")
print("sample shape:", true_diff_params["sample"].shape)
print("probe_modes shape:", true_diff_params["probe_modes"].shape)


# %%
# 2) Simulate a forward pass to get synthetic data
mask = jnp.ones((forward_cfg.probe_size, forward_cfg.probe_size), dtype=jnp.bool_)
pos_idx = jnp.arange(forward_cfg.n_positions, dtype=jnp.int32)
batched_model = jax.jit(jax.vmap(point_model, in_axes=(None, None, 0)))
true_intensity = batched_model(true_diff_params, true_non_diff_params, pos_idx)

if forward_cfg.measurement_noise_std > 0:
    meas_noise = rng_forward.normal(
        0.0,
        forward_cfg.measurement_noise_std,
        size=true_intensity.shape,
    ).astype(np.float32)
    true_intensity = jnp.clip(true_intensity + meas_noise, a_min=0.0)

measured_root_intensity = jnp.sqrt(true_intensity + 1e-20)
print("measured_root_intensity shape:", measured_root_intensity.shape)


# %%
# 3) Reconstruction config
recon_cfg = ReconstructionConfig(
    reconstruction_type="accumulate_full_pass",  # accumulate_full_pass | sequential_no_repeats | random_with_replacement
    n_steps=300,
    batch_size=50,
    shuffle_each_epoch=True,
    use_multigpu=False,
    seed=1,
    position_noise_px=2,
    init_scan_mistake_std=0.3,
    init_probe_noise_std=20.0,
    tv_sample_weight=0.0,
    tv_probe_modes_weight=0.0,
    shift_rounding_step=3.0,
    max_allowed_whole_shift=10.0,
)

print("reconstruction mode:", recon_cfg.reconstruction_type)
print("n_steps:", recon_cfg.n_steps, "batch_size:", recon_cfg.batch_size)


# %%
# 4) Create reconstruction model/loss/optimizer/projections
rng_recon = np.random.default_rng(recon_cfg.seed)
n_positions = measured_root_intensity.shape[0]

params = {
    "sample": jnp.ones_like(true_diff_params["sample"]),
    "probe_modes": jnp.asarray(true_diff_params["probe_modes"]),
    "modal_weights": jnp.ones_like(true_diff_params["modal_weights"]),
    "scan_mistakes": (
        jnp.zeros((n_positions, 2), dtype=jnp.float32)
        + jnp.asarray(
            rng_recon.normal(
                0.0, recon_cfg.init_scan_mistake_std, size=(n_positions, 2)
            ),
            dtype=jnp.float32,
        )
    ),
}

if recon_cfg.init_probe_noise_std > 0:
    probe_noise = jnp.asarray(
        rng_recon.normal(
            0.0, recon_cfg.init_probe_noise_std, size=params["probe_modes"].shape
        ),
        dtype=jnp.float32,
    )
    params["probe_modes"] = params["probe_modes"] + (probe_noise - probe_noise.mean())

scan_pos_noise = rng_recon.integers(
    -recon_cfg.position_noise_px,
    recon_cfg.position_noise_px + 1,
    size=true_non_diff_params["sample_positions"].shape,
    endpoint=False,
)
non_diff_params = {
    "sample_positions": jnp.asarray(
        np.asarray(true_non_diff_params["sample_positions"]) + scan_pos_noise,
        dtype=jnp.int32,
    ),
    "tv_sample_weight": recon_cfg.tv_sample_weight,
    "tv_probe_modes_weight": recon_cfg.tv_probe_modes_weight,
}

data_loss = jax.jit(join_loss_and_forward_batched(gauss_loss, point_model))
reg_sample = create_regularizer(
    diff_params_name="sample",
    reg_weight_name="tv_sample_weight",
    reg_func=create_tv_reg(p=2, q=2, conv_func=jnp.angle),
)
reg_probe = create_regularizer(
    diff_params_name="probe_modes",
    reg_weight_name="tv_probe_modes_weight",
    reg_func=create_tv_reg(p=2, q=2, conv_func=jnp.abs),
)
regularizer = combine_regularizers([reg_sample, reg_probe])
objective = join_loss_and_reg(data_loss, regularizer)
loss_and_grad_fn = jax.jit(jax.value_and_grad(objective, argnums=0))

opt_params_per_leaf = {
    "sample": {"type": "adam", "learning_rate": 1e-1},
    "probe_modes": {"type": "adam", "learning_rate": 1e0},
    "modal_weights": {"type": "adam", "learning_rate": 0.0},
    "scan_mistakes": {"type": "adam", "learning_rate": 1e-2},
}
optimizer = prepare_optimizer(opt_params_per_leaf)
opt_state = optimizer.init(params)

proj_shift_rounder = pj.create_projection_applier(
    pj.create_shift_rounder(
        max_correction_magnitude=forward_cfg.max_correction_magnitude,
        rounding_step=recon_cfg.shift_rounding_step,
    ),
    diff_param_name="scan_mistakes",
    non_diff_param_name="sample_positions",
)
proj_scan_limit = pj.create_projection_applier(
    pj.create_shift_limiter(
        max_allowed_shift=recon_cfg.max_allowed_whole_shift,
        initial_positions=non_diff_params["sample_positions"],
    ),
    non_diff_param_name="sample_positions",
)
proj_sample_clip = pj.create_projection_applier(
    pj.create_modulus_clipper(max_mod_val=1.0, min_mod_val=0.0),
    diff_param_name="sample",
)
projection_fn = pj.merge_multiple_projections(
    [proj_shift_rounder, proj_scan_limit, proj_sample_clip]
)


# %%
# 5) Reconstruction run
params, opt_state, loss_hist = run_optimization_loop(
    params=params,
    opt_state=opt_state,
    optimizer=optimizer,
    loss_and_grad_fn=loss_and_grad_fn,
    non_diff_params=non_diff_params,
    measured_batch_pool=measured_root_intensity,
    mask=mask,
    mode=recon_cfg.reconstruction_type,
    n_steps=recon_cfg.n_steps,
    batch_size=recon_cfg.batch_size,
    seed=recon_cfg.seed,
    shuffle_each_epoch=recon_cfg.shuffle_each_epoch,
    use_multigpu=recon_cfg.use_multigpu,
    projection_fn=projection_fn,
)

print(f"Final loss: {float(loss_hist[-1]):.6e}")


# %%
# 6) Result display
plt.figure(figsize=(6, 3))
plt.plot(loss_hist)
plt.title(f"Loss history ({recon_cfg.reconstruction_type})")
plt.xlabel("step")
plt.ylabel("loss")
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(jnp.abs(params["sample"]), cmap="turbo")
plt.title("Reconstructed |sample|")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(jnp.angle(params["sample"]), cmap="twilight")
plt.title("Reconstructed angle(sample)")
plt.axis("off")
plt.tight_layout()

probe_modes = params["probe_modes"]
if probe_modes.ndim == 2:
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(jnp.abs(probe_modes), cmap="viridis")
    plt.title("|probe|")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(jnp.angle(probe_modes), cmap="twilight")
    plt.title("angle(probe)")
    plt.axis("off")
    plt.tight_layout()
else:
    n_modes = probe_modes.shape[0]
    plt.figure(figsize=(4 * n_modes, 6))
    for i in range(n_modes):
        plt.subplot(2, n_modes, i + 1)
        plt.imshow(jnp.abs(probe_modes[i]), cmap="viridis")
        plt.title(f"Mode {i + 1} |.|")
        plt.axis("off")

        plt.subplot(2, n_modes, n_modes + i + 1)
        plt.imshow(jnp.angle(probe_modes[i]), cmap="twilight")
        plt.title(f"Mode {i + 1} angle")
        plt.axis("off")
    plt.tight_layout()

plt.show()

# %%
