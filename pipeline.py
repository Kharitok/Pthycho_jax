# %% reconstruction pipeline for ptychography

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"
import time

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dataloader import (
    get_reconstruction_resolution,
    load_dataset,
    load_exp_params_from_attributes,
    process_scan_coordinates,
)
from probe_initialization import get_thresholded_intensity, init_probe_simple_fft

# %%

dataloader_config = {
    "file_path": "/gpfs/exfel/exp/MID/202425/p008476/scratch/Kostya/Data/Aligned_ds/r_474_473/r_474_473_ptycho_data.h5",
    "scan_coord_paths": ("diff_position_x", "diff_position_y"),
    "calculate_avg_image": True,
    "avg_image_path": "average_image",
    "load_probe_image": True,
    "probe_image_path": "avg_beam",
    "calculate_sortings": True,
    "save_sortings": True,
    "mask_path": "mask",
    "precise_mask_path": "precise_mask",
    "scan_coord_limits": {
        "0": (-4, -3),
        "1": (4, 6),
    },
    "intensity_limits": (1e3, 1e9),
}

experimental_parameters = (
    "detector_pixel_size_m",
    "pixel_number",
    "run_number",
    "sample_detector_distance_m",
    "wavelength_m",
)


reconstruction_config = {
    "scan_motors_unit": 1e-3,
    "propogation_function": "fft",
    "assumed_defocus_at_sample_m": -15.9e-3,
    "relative_threshold_for_probe_estimation": 5e-3,
    "probe_modes_num": 1,
    "bin_factor": (1, 1),
    "shift_margin": 10,
    "max_correction_magnitude": 5,
}


# %% read data
# loader already sorts measured points according to the scan coordinates and intensities


loaded_data = load_dataset(dataloader_config)
loaded_parameters = load_exp_params_from_attributes(
    dataloader_config["file_path"], experimental_parameters
)

get_non_neggative_non_nans = lambda x: np.maximum(np.nan_to_num(x), 0)

loaded_data["measured_intensities"] = get_non_neggative_non_nans(
    loaded_data["measured_intensities"]
)
loaded_data["mean_image"] = get_non_neggative_non_nans(loaded_data["mean_image"])
# TODO patch loading
loaded_data["precise_mask"] = (
    loaded_data["precise_mask"].T if loaded_data["precise_mask"] is not None else None
)


# %% process scan coordinates
# convert to pixes, center, split into int and float parts


resolution = get_reconstruction_resolution(
    loaded_parameters["detector_pixel_size_m"],
    loaded_parameters["pixel_number"],
    loaded_parameters["sample_detector_distance_m"],
    loaded_parameters["wavelength_m"],
)
reconstruction_pix_size = resolution[0]

print(f"Reconstruction resolution: {reconstruction_pix_size/1e-9:.1f} nm")


int_parts, float_parts = process_scan_coordinates(
    loaded_data["scan_coords"], reconstruction_pix_size
)

plt.figure()
plt.scatter(int_parts[:, 0], int_parts[:, 1])
plt.title("Integer parts of scan coordinates in pixels")
plt.show()

### Get assumption of the sample size in pixels

min_required_sample_size_pix = (
    (np.max(int_parts, axis=0) - np.min(int_parts, axis=0))
    + 1
    + np.array(loaded_data["mean_probe"].shape)
)
sample_size_pix = (min_required_sample_size_pix * 1.2).astype(int)
### get beam estimation from distance and defocus
# %%


# def get_thresholded_intensity(
#     I_det: np.ndarray, relative_threshold: float = 1e-2
# ) -> np.ndarray:
#     """
#     Computes the thresholded intensity of the probe from the measured intensity at the detector.

#     Parameters:
#     -----------
#     I_det : 2D array -> Measured empty beam intensity at detector
#     relative_threshold : float -> Relative threshold for probe estimation

#     Returns:
#     --------
#     detector_probe_modulus : 2D array -> Thresholded modulus of the probe at the detector
#     """
#     # Ensure non-negative intensities
#     I_det = np.maximum(np.nan_to_num(I_det, nan=0.0), 0)

#     # Create a support mask based on the relative threshold
#     detector_probe_support = I_det > np.max(I_det) * relative_threshold

#     # Compute the modulus of the probe
#     detector_probe_modulus = I_det * detector_probe_support

#     return detector_probe_modulus


probe_intensity = get_thresholded_intensity(
    loaded_data["mean_probe"],
    reconstruction_config["relative_threshold_for_probe_estimation"],
)

probe = init_probe_simple_fft(
    probe_intensity,
    loaded_parameters["detector_pixel_size_m"],
    loaded_parameters["sample_detector_distance_m"],
    -reconstruction_config["assumed_defocus_at_sample_m"],
    loaded_parameters["wavelength_m"],
)[0]


plt.figure()
plt.imshow(np.abs(probe), norm="asinh")
plt.title("Initial probe modulus at sample plane")
plt.show()


# %%### build modes and modal weights
n_modes = 2  # reconstruction_config["probe_modes_num"]

modes_wavefields = np.array([probe for _ in range(n_modes)])
modes_wavefields += 1e-2 * np.random.randn(*modes_wavefields.shape)
plt.imshow(np.abs(modes_wavefields[0]))
plt.title("Initial mode wavefield (first mode)")
plt.show()


# normalize the modes to match the mean measured intensity
modes_energy = (np.abs(modes_wavefields) ** 2).sum(axis=(-1, -2))

mean_measured_intensity = loaded_data["total_ints"].mean()

scaling_factor = mean_measured_intensity / modes_energy.sum()
modes_wavefields *= np.sqrt(scaling_factor)

# normalize modal weights to match exact intensity

modal_weights = np.diag(np.ones(n_modes))
modal_weights = np.array(
    [modal_weights for _ in range(loaded_data["total_ints"].shape[0])]
)
modal_weights_scaling = loaded_data["total_ints"] / mean_measured_intensity
modal_weights *= modal_weights_scaling[:, None, None]


# %% Probe mask, freq_mask, detector mask

# TODO add mask constructions based on configs
real_mask = np.ones_like(probe_intensity, dtype=bool)
freq_mask = probe_intensity > np.nanmax(probe_intensity) * 1e-3
detector_mask = (
    loaded_data["mask"] * loaded_data["precise_mask"]
    if loaded_data["precise_mask"] is not None
    else loaded_data["mask"]
)
# %% ifftshift all for reconstruction
loaded_data["measured_intensities"] = np.fft.ifftshift(
    loaded_data["measured_intensities"], axes=(-2, -1)
)
loaded_data["mean_image"] = np.fft.ifftshift(loaded_data["mean_image"])
detector_mask = np.fft.ifftshift(detector_mask)
freq_mask = np.fft.ifftshift(freq_mask)
print("Loaded data and masks have been ifftshifted.")
# %% Show all befor the reconstruction
# probe
plt.figure()
for i in range(n_modes):
    # top amlitude bottom phase
    for j in range(2):
        plt.subplot(2, n_modes, j * n_modes + i + 1)
        if j == 0:
            plt.imshow(np.abs(modes_wavefields[i]), cmap="turbo", norm="asinh")
            plt.title(f"Mode {i + 1} amplitude")
            plt.colorbar()
        else:
            plt.imshow(np.angle(modes_wavefields[i]), cmap="twilight")
            plt.title(f"Mode {i + 1} phase")
plt.suptitle("Probe modes")
plt.tight_layout()
plt.show()

# probe at detector
modes_wavefields_detector = np.fft.fft2(modes_wavefields, axes=(-2, -1))

plt.figure()
for i in range(n_modes):
    # top amlitude bottom phase
    plt.subplot(1, n_modes, i + 1)
    plt.imshow(np.abs(modes_wavefields_detector[i]), cmap="turbo", norm="asinh")
    plt.title(f"Detector {i + 1} amplitude")
    plt.colorbar()
plt.suptitle("Probe modes at detector")
plt.tight_layout()
plt.show()

# masks and measured data
plt.figure(figsize=(10, 40))
plt.subplot(1, 5, 1)
plt.imshow(real_mask, cmap="gray")
plt.title("Real mask")
plt.subplot(1, 5, 2)
plt.imshow(freq_mask, cmap="gray")
plt.title("Frequency mask")
plt.subplot(1, 5, 3)
plt.imshow(detector_mask, cmap="gray")
plt.title("Detector mask")
plt.subplot(1, 5, 4)
plt.imshow(loaded_data["mean_image"], cmap="turbo", norm="asinh")
plt.title("Measured intensity")
plt.subplot(1, 5, 5)
plt.imshow(loaded_data["measured_intensities"][0], cmap="turbo", norm="asinh")
plt.title("First measured")

plt.tight_layout()
plt.show()

# %%
reconstruction_config = {
    "scan_motors_unit": 1e-3,
    "propogation_function": "fft",
    "assumed_defocus_at_sample_m": -15.9e-3,
    "relative_threshold_for_probe_estimation": 5e-3,
    "probe_modes_num": 1,
    "bin_factor": (1, 1),
    "shift_margin": 10,
    "max_correction_magnitude": 5,
}
reconstruction_config["probe_size"] = loaded_data["mean_image"].shape

from loss_and_reg import (
    combine_regularizers,
    create_regularizer,
    create_tv_reg,
    gauss_loss,
    join_loss_and_forward,
    join_loss_and_forward_batched,
    join_loss_and_reg,
)
from optim_loop import run_optimization_loop
from optimizers import prepare_optimizer

# %% Here we should have all data loaded and prepared
# Now it's time for the model constructions
from transmission_ptycho_models import construct_point_model_ptycho_transmission

# %% construct model


model_parameters = {
    "probe_type": "fluctuating",
    "sample_type": "complex",
    "binning_factor": (1, 1),
    "shift_margin": reconstruction_config["shift_margin"],
    "sample_selector_type": "correcting",
    "propagator_type": "Fourrier",
    "probe_shape": reconstruction_config["probe_size"],
    "max_correction_magnitude": 10,
}

models = construct_point_model_ptycho_transmission(model_parameters)


# %% construct optimizable and fixed parameters

n_pos = 1000
# TODO here float point for shifts should be different and recaclulated from shifts
differentiable_parameters = {
    "sample": jnp.ones(sample_size_pix).astype(jnp.complex64),
    "probe_modes": jnp.array(modes_wavefields.copy()).astype(jnp.complex64),
    "modal_weights": jnp.array(modal_weights.copy()).astype(jnp.complex64),
    "scan_mistakes": jnp.array(np.zeros_like(float_parts)).astype(jnp.float32),
}
non_differentiable_parameters = {
    "sample_positions": jnp.array(int_parts - np.min(int_parts, axis=0) + 50).astype(
        jnp.int32
    ),
}


batch_measured = jnp.sqrt(loaded_data["measured_intensities"][:n_pos])

# %% construct loss projectors and optimizers

# loss

get_loss_v = jax.jit(join_loss_and_forward_batched(gauss_loss, models["point_model"]))
get_loss_and_grad_v = jax.jit(jax.value_and_grad(get_loss_v, argnums=0))

# %%optimizer
opt_params_per_leaf = {
    "sample": {"type": "adam", "learning_rate": 1e0},  # 1e0
    "probe_modes": {"type": "adam", "learning_rate": 1e0},  # 1e0
    "modal_weights": {"type": "adam", "learning_rate": 1e-1},  # 1e-3
    "scan_mistakes": {"type": "adam", "learning_rate": 1e-2},  # 1e-2
}

optimizer = prepare_optimizer(opt_params_per_leaf)
opt_state = optimizer.init(differentiable_parameters)

# %% show before

plt.subplot(1, 2, 1)
plt.imshow(
    jnp.abs(differentiable_parameters["sample"]), cmap="turbo"
)  # [30:100,40:110]
plt.colorbar()
# plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(jnp.angle(differentiable_parameters["sample"]), cmap="turbo")
# switch off axis
# plt.axis("off")
plt.tight_layout()
plt.show()
plt.figure()
if differentiable_parameters["probe_modes"].ndim < 3:
    plt.figure()
    # show abs and angle of the sample
    plt.subplot(1, 2, 1)
    plt.imshow(
        jnp.abs(differentiable_parameters["probe_modes"]), cmap="turbo"
    )  # [30:100,40:110]
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(jnp.angle(differentiable_parameters["probe_modes"]), cmap="turbo")
    # switch off axis
    plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    n_modes = differentiable_parameters["probe_modes"].shape[0]
    plt.figure(figsize=(12, 4))
    for i in range(n_modes):
        plt.subplot(2, n_modes, i + 1)
        plt.imshow(jnp.abs(differentiable_parameters["probe_modes"][i]), cmap="viridis")
        plt.axis("off")
        plt.title(f"Mode {i+1} Abs")
        plt.subplot(2, n_modes, n_modes + i + 1)
        plt.imshow(
            jnp.angle(differentiable_parameters["probe_modes"][i]), cmap="twilight"
        )
        plt.axis("off")
        plt.title(f"Mode {i+1} Angle")
    plt.tight_layout()
    plt.show()


# %% Start recon loop
# n_pos = 200
time_0 = time.time()

differentiable_parameters, non_differentiable_parameters, opt_state, loss_hist = (
    run_optimization_loop(
        params=differentiable_parameters,
        non_diff_params=non_differentiable_parameters,
        opt_state=opt_state,
        optimizer=optimizer,
        loss_and_grad_fn=get_loss_and_grad_v,
        measured_batch_pool=batch_measured[:n_pos],
        mask=jnp.array(detector_mask).astype(bool),
        mode='accumulate_full_pass_streaming',#"accumulate_full_pass",
        n_steps=250,  # epochs
        batch_size=100,  # memory-fit batch
        seed=0,
        shuffle_each_epoch=True,
        projection_fn=lambda x, y: (x, y),
        use_multigpu=True,
    )
)
time_1 = time.time()
print(
    f"Optimization took {int(time_1 - time_0)} seconds. {(time_1-time_0)/250} per step"
)

plt.figure()
plt.plot(loss_hist)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Optimization Loss History")
plt.show()

# %% show after

plt.subplot(1, 2, 1)
plt.imshow(
    jnp.abs(differentiable_parameters["sample"])[200:1000, 100:1000], cmap="turbo"
)  # [30:100,40:110]
plt.colorbar()
# plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(
    jnp.angle(differentiable_parameters["sample"])[200:1000, 100:1000], cmap="turbo"
)
# switch off axis
# plt.axis("off")
plt.tight_layout()
plt.show()
plt.figure()
if differentiable_parameters["probe_modes"].ndim < 3:
    plt.figure()
    # show abs and angle of the sample
    plt.subplot(1, 2, 1)
    plt.imshow(
        jnp.abs(differentiable_parameters["probe_modes"]), cmap="turbo"
    )  # [30:100,40:110]
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(jnp.angle(differentiable_parameters["probe_modes"]), cmap="turbo")
    # switch off axis
    plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    n_modes = differentiable_parameters["probe_modes"].shape[0]
    plt.figure(figsize=(12, 4))
    for i in range(n_modes):
        plt.subplot(2, n_modes, i + 1)
        plt.imshow(jnp.abs(differentiable_parameters["probe_modes"][i]), cmap="viridis")
        plt.axis("off")
        plt.title(f"Mode {i+1} Abs")
        plt.subplot(2, n_modes, n_modes + i + 1)
        plt.imshow(
            jnp.angle(differentiable_parameters["probe_modes"][i]), cmap="twilight"
        )
        plt.axis("off")
        plt.title(f"Mode {i+1} Angle")
    plt.tight_layout()
    plt.show()

# %%
