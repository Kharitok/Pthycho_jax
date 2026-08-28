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
}


# %% read data
# loader already sorts measured points according to the scan coordinates and intensities


loaded_data = load_dataset(dataloader_config)
loaded_parameters = load_exp_params_from_attributes(
    dataloader_config["file_path"], experimental_parameters
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
### build modes and modal weights

# %%
