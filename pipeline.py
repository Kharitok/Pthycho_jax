# %% reconstruction pipeline for ptychography
import h5py
import matplotlib.pyplot as plt
import numpy as np

from dataloader import (
    get_reconstruction_resolution,
    load_dataset,
    load_exp_params_from_attributes,
    process_scan_coordinates,
)
from probe_initialization import init_probe_simple_fft

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


# def process_scan_coordinates(
#     scan_coords: np.ndarray,
#     reconstruction_pix_size: float,
#     scan_coords_unit: float = 1e-3,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Processes the scan coordinates by converting them to pixels, centering them, and splitting into integer and float parts.

#     Args:
#         scan_coords (np.ndarray): The scan coordinates in the unit specified by `scan_coords_unit`.
#         reconstruction_pix_size (float): The pixel size for reconstruction in meters.
#         scan_coords_unit (float): The unit of the scan coordinates (default is 1e-3 for millimeters).

#     Returns:
#         tuple: A tuple containing the integer and float parts of the processed scan coordinates.
#     """
#     # Convert to pixels
#     scan_coords_pix = scan_coords * scan_coords_unit / reconstruction_pix_size

#     # Center the coordinates
#     scan_coords_centered = scan_coords_pix - np.mean(scan_coords_pix, axis=0)

#     # Split into integer and float parts
#     int_parts = np.floor(scan_coords_centered).astype(int)
#     float_parts = scan_coords_centered - int_parts

#     return int_parts, float_parts


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


def get_thresholded_intensity(
    I_det: np.ndarray, relative_threshold: float = 1e-2
) -> np.ndarray:
    """
    Computes the thresholded intensity of the probe from the measured intensity at the detector.

    Parameters:
    -----------
    I_det : 2D array -> Measured empty beam intensity at detector
    relative_threshold : float -> Relative threshold for probe estimation

    Returns:
    --------
    detector_probe_modulus : 2D array -> Thresholded modulus of the probe at the detector
    """
    # Ensure non-negative intensities
    I_det = np.maximum(np.nan_to_num(I_det, nan=0.0), 0)

    # Create a support mask based on the relative threshold
    detector_probe_support = I_det > np.max(I_det) * relative_threshold

    # Compute the modulus of the probe
    detector_probe_modulus = I_det * detector_probe_support

    return detector_probe_modulus


probe_intensity = get_thresholded_intensity(
    loaded_data["mean_probe"],
    reconstruction_config["relative_threshold_for_probe_estimation"],
)

probe = init_probe_simple_fft(
    probe_intensity,
    loaded_parameters["detector_pixel_size_m"],
    loaded_parameters["sample_detector_distance_m"],
    reconstruction_config["assumed_defocus_at_sample_m"],
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


# %% Here we should have all data loaded and prepared
# Now it's time for the model constructions
