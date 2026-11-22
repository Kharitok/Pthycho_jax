# %% reconstruction pipeline for ptychography

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
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
    "batch_size": 100,
    "optim_steps": 100,
    "use_positions_in_reconstruction": 1000,
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
n_modes = 5  # reconstruction_config["probe_modes_num"]

modes_wavefields = np.array([probe for _ in range(n_modes)])
modes_wavefields += 1e-2 * np.random.randn(*modes_wavefields.shape)

# modes_wavefields[1] = np.roll(modes_wavefields[1], shift=1, axis=-1)
# modes_wavefields[1] = np.roll(modes_wavefields[1], shift=1, axis=-2)
# modes_wavefields[1] = np.roll(modes_wavefields[2], shift=1, axis=-1)
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

noise_scale = 1e-3
amp_noise = np.random.randn(*modal_weights.shape)
phase_noise = np.random.uniform(-np.pi, np.pi, modal_weights.shape)
modal_weights = modal_weights + noise_scale * amp_noise * np.exp(1j * phase_noise)


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
    "sample_type": "complex",#"modulus_phase",  # "complex",
    "binning_factor": (1, 1),
    "shift_margin": reconstruction_config["shift_margin"],
    "sample_selector_type": "correcting",
    "propagator_type": "Fourrier",
    "probe_shape": reconstruction_config["probe_size"],
    "max_correction_magnitude": 10,
    "shift_margin": 5,
}

models = construct_point_model_ptycho_transmission(model_parameters)


# %% construct optimizable and fixed parameters
# 9493
n_pos = 9000
# TODO here float point for shifts should be different and recaclulated from shifts
differentiable_parameters = {
    "sample": jnp.ones(sample_size_pix).astype(jnp.complex64),
    # "sample_modulus_logit": jnp.array(
    # jax.scipy.special.logit(jnp.zeros(sample_size_pix) + 0.9)
    # ).astype(jnp.float32),
    # "sample_phase": jnp.zeros(sample_size_pix).astype(jnp.float32),
    "probe_modes": jnp.array(modes_wavefields.copy()).astype(jnp.complex64),
    "modal_weights": jnp.array(modal_weights.copy()).astype(jnp.complex64),
    "scan_mistakes": jnp.array(
        np.arctanh(float_parts / reconstruction_config["max_correction_magnitude"])
    ).astype(jnp.float32),
}
non_differentiable_parameters = {
    "sample_positions": jnp.array(int_parts - np.min(int_parts, axis=0) + 50).astype(
        jnp.int32
    ),
}


batch_measured = np.sqrt(loaded_data["measured_intensities"][:n_pos])

# %% construct loss projectors and optimizers

# regularization

# tv_reg = create_tv_reg(weight=1e-4)  # Example total variation regularization


# loss

get_loss_v = jax.jit(join_loss_and_forward_batched(gauss_loss, models["point_model"]))
get_loss_and_grad_v = jax.jit(jax.value_and_grad(get_loss_v, argnums=0))



# %%optimizer
opt_params_per_leaf = {
    "sample": {"type": "adam", "learning_rate": 1e0},  # 1e0
    # "sample_modulus_logit": {"type": "adam", "learning_rate": 1e0},
    # "sample_phase": {"type": "adam", "learning_rate": 1e0},
    "probe_modes": {"type": "adam", "learning_rate": 1e0},  # 1e0
    "modal_weights": {"type": "adam", "learning_rate": 1e-1},  # 1e-3
    "scan_mistakes": {"type": "adam", "learning_rate": 1e-2},  # 1e-2
}

optimizer = prepare_optimizer(opt_params_per_leaf)
opt_state = optimizer.init(differentiable_parameters)

# %% show before


if "sample" in differentiable_parameters:
    sample = differentiable_parameters["sample"]
else:
    sample_modulus_logit = differentiable_parameters["sample_modulus_logit"]
    sample_phase = differentiable_parameters["sample_phase"]
    sample = jax.nn.sigmoid(sample_modulus_logit) * jnp.exp(1j * sample_phase)
plt.subplot(1, 2, 1)
plt.imshow(jnp.abs(sample), cmap="turbo")  # [30:100,40:110]
plt.colorbar()
# plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(jnp.angle(sample), cmap="turbo")
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
# n_pos = 1200
batch_size = 600
n_steps = 400
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
        mode="accumulate_full_pass",  # "accumulate_full_pass_streaming",  # "accumulate_full_pass",
        n_steps=n_steps,  # epochs
        batch_size=batch_size,  # memory-fit batch
        seed=0,
        shuffle_each_epoch=True,
        projection_fn=lambda x, y: (x, y),
        use_multigpu=True,
    )
)
time_1 = time.time()
print(
    f"Optimization took {int(time_1 - time_0)} seconds. {(time_1-time_0)/n_steps} per step"
)

plt.figure()
plt.plot(loss_hist)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Optimization Loss History")
plt.show()

# %% show after
if "sample" in differentiable_parameters:
    sample = differentiable_parameters["sample"]
else:
    sample_modulus_logit = differentiable_parameters["sample_modulus_logit"]
    sample_phase = differentiable_parameters["sample_phase"]
    sample = jax.nn.sigmoid(sample_modulus_logit) * jnp.exp(1j * sample_phase)
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.imshow(jnp.abs(sample), cmap="turbo")  # [30:100,40:110]
plt.colorbar()
# plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(jnp.angle(sample), cmap="turbo")
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



# dataloadin nanomax


datapath = '/gpfs/exfel/data/scratch/kkonstantin/Projects/NanoMax/Test_reconstruction/download_data/09_kev/raw/0001_setup/000082.h5'

# #print key, shape  for datasets for each group recursevely as tree
# def print_h5_tree(name, obj):
#     if isinstance(obj, h5py.Dataset):
#         print(f"{name}: {obj.shape}")

# with h5py.File(datapath, "r") as f:
#     f.visititems(print_h5_tree)



from pathlib import Path

import h5py
import hdf5plugin


class H5DataLoader:
    def __init__(self, filepath: str | Path, mode: str = "r") -> None:
        self.filepath = Path(filepath)
        self.mode = mode
        self._file: h5py.File | None = None

    def __enter__(self) -> "H5DataLoader":
        if self._file is not None:
            raise RuntimeError("H5DataLoader context is already open.")
        self._file = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError(
                "Cannot access 'self.file' outside of a 'with' block."
            )
        return self._file

    def read_dataset(self, key: str):
        return self.file[key][:]


class NanoMaxDataset:

    detector_path: str = "/entry/measurement/eiger4m/frames"
    scan_x_path: str = "/entry/measurement/sx"
    scan_y_path: str = "/entry/measurement/sy"
    # detector_distance_path: str = "/entry/measurement/detector_distance"
    energy_path: str = "/entry/snapshots/pre_scan/energy_raw"
    # defocus_path: str = "/entry/measurement/defocus"




    def __init__(self, datapath: str | Path,detector_distance:float,detector_pixel_size:float,defocus:float) -> None:
        self.datapath = Path(datapath)
        print(f"Initialized NanoMaxDataset with datapath: {self.datapath}")
        self.detector_distance = detector_distance
        self.detector_pixel_size = detector_pixel_size
        self.defocus = defocus


    def __getitem__(self, key: str):
        # Context manager lives inside the class; file closes automatically on exit
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(key)

    @property
    def detector_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.detector_path)
    
    @property
    def scan_x_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.scan_x_path)

    @property
    def scan_y_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.scan_y_path)

    # @property
    # def detector_distance_data(self) -> h5py.File:
    #     with H5DataLoader(self.datapath) as loader:
    #         return loader.read_dataset(self.detector_distance_path)

    @property
    def energy_data(self) -> h5py.File:
        with H5DataLoader(self.datapath) as loader:
            return loader.read_dataset(self.energy_path)

    # @property
    # def defocus_data(self) -> h5py.File:
    #     with H5DataLoader(self.datapath) as loader:
    #         return loader.read_dataset(self.defocus_path)

    @property
    def detector_mask(self) -> h5py.File:
        measured_data = self.detector_data
        return measured_data[0] <measured_data[0].max()






dataset = NanoMaxDataset(datapath,detector_distance = 7.180,
                         detector_pixel_size = 75e-6,
                         defocus = 1500e-6)


#%%


detector_data = dataset.detector_data
mask = dataset.detector_mask

mean_data = (detector_data * mask).mean(axis=0)

plt.figure()
plt.imshow(mean_data , norm='asinh')
plt.colorbar()

from scipy.ndimage import center_of_mass

com = center_of_mass(mean_data)
patch = mean_data[int(com[0])-200:int(com[0])+200, int(com[1])-200:int(com[1])+200]
com_patch = center_of_mass(patch>patch.max()*0.1)

com_all = (com[0] - 200 + com_patch[0], com[1] - 200 + com_patch[1])


print(f"Center of mass of the mean detector data: {com}")
print(f"Refined center of mass from patch: {com_all}")
plt.figure()
plt.imshow(mean_data, norm='log')
plt.scatter(com[1], com[0], color='red', marker='x')
plt.scatter(com_all[1], com_all[0], color='blue', marker='o')
plt.title("Mean Detector Data with Center of Mass")
plt.colorbar()

# %%


plt.figure()
plt.imshow(patch, )
plt.scatter(com_patch[1], com_patch[0], color='red', marker='x')
plt.title("Refined Patch with Center of Mass")
plt.colorbar()

# %%
