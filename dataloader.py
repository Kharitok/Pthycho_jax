# %%

from typing import Tuple

import h5py
import numpy as np

# %%


def read_scan_coordinates_from_h5(
    h5_file_path: str, scan_cords_pathes: Tuple[str, str]
) -> np.ndarray:
    """
    Reads scan coordinates from an HDF5 file.

    Args:
        h5_file_path (str): Path to the HDF5 file.
        scan_cords_pathes (Tuple[str, str]): Tuple containing the paths to the scan coordinates datasets within the HDF5 file.

    Returns:
        np.ndarray: Array containing the scan coordinates.
    """

    if len(scan_cords_pathes) != 2:
        raise ValueError(
            "scan_cords_pathes must be a tuple of two strings for two scan coordinates"
        )

    with h5py.File(h5_file_path, "r") as f:
        scan_coordinates = np.stack([f[path][:] for path in scan_cords_pathes], axis=-1)
    return scan_coordinates


def get_unique_positions(
    scan_coordinates: np.ndarray, axis: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns unique positions from the scan coordinates.

    Args:
        scan_coordinates (np.ndarray): Array containing the scan coordinates.
        axis (int, optional): Axis along which to find unique positions. Defaults to 0.
    """

    unique_pairs, first_idx, inverse_idx, counts = np.unique(
        scan_coordinates,
        axis=axis,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    return unique_pairs, first_idx, inverse_idx, counts


def get_sorting_indicies(inverse_idx: np.ndarray, total_int: np.ndarray) -> np.ndarray:
    """
    Returns the sorting indices based on position and intensity.

    Args:
        inverse_idx (np.ndarray): Inverse indices of unique positions.
        total_int (np.ndarray): Total intensities of the images.

    Returns:
        np.ndarray: Sorting indices based on position and intensity.
    """
    pos_id = inverse_idx
    intensity = total_int

    group_order = np.lexsort(
        (-intensity, pos_id)
    )  # group by pos_id, intensity desc within group
    sorted_pos = pos_id[group_order]
    rank = np.arange(len(pos_id)) - np.searchsorted(sorted_pos, sorted_pos, side="left")

    final_idx = group_order[np.lexsort((sorted_pos, rank))]
    return final_idx


def get_im_sums(filepath, image_path):
    """
    Reads an image from an HDF5 file and calculates the sum of its pixel values.

    Args:
        filepath (str): Path to the HDF5 file.
        image_path (str): Path to the image dataset within the HDF5 file.

    Returns:
        float: Sums of the pixel values of the images.
    """
    with h5py.File(filepath, "r") as f:
        image = f[image_path][:]
    return np.nansum(image, axis=(-1, -2))


def get_images(filepath, image_path) -> np.ndarray:
    """
    Reads an image from an HDF5 file.

    Args:
        filepath (str): Path to the HDF5 file.
        image_path (str): Path to the image dataset within the HDF5 file.
    Returns:
        np.ndarray: Array containing the images.
    """
    with h5py.File(filepath, "r") as f:
        image = f[image_path][:]
    return image


def get_validity_mask_for_coords(
    lim_0: tuple[float, float], lim_1: tuple[float, float], scan_coords: np.ndarray
) -> np.ndarray:
    """
    Returns a boolean mask indicating which scan coordinates are within the specified limits.

    Args:
        lim_0 (tuple): Tuple containing the lower and upper limits for the first coordinate.
        lim_1 (tuple): Tuple containing the lower and upper limits for the second coordinate.
        scan_coords (np.ndarray): Array containing the scan coordinates.

    Returns:
        np.ndarray: Boolean mask indicating valid scan coordinates.
    """
    return (
        (scan_coords[:, 0] >= lim_0[0])
        & (scan_coords[:, 0] <= lim_0[1])
        & (scan_coords[:, 1] >= lim_1[0])
        & (scan_coords[:, 1] <= lim_1[1])
    )


def get_validity_mask_for_intensities(
    lim: tuple[float, float], total_int: np.ndarray
) -> np.ndarray:
    """
    Returns a boolean mask indicating which total intensities are within the specified limits.

    Args:
        lim (tuple): Tuple containing the lower and upper limits for the total intensities.
        total_int (np.ndarray): Array containing the total intensities.

    Returns:
        np.ndarray: Boolean mask indicating valid total intensities.
    """
    return (total_int >= lim[0]) & (total_int <= lim[1])


# %%


# load img sums
# get sorting index
# get uniqueness index
# for position in unique positions
t_path = "/gpfs/exfel/exp/MID/202425/p008476/scratch/Kostya/Data/Aligned_ds/r_474_473/r_474_473_ptycho_data.h5"


cord_x, cord_y = "diff_position_x", "diff_position_y"
intensities = "images"


# %%

scan_coords = read_scan_coordinates_from_h5(t_path, (cord_x, cord_y))
unique, first_idx, inverse_idx, counts = get_unique_positions(scan_coords)

total_int = get_im_sums(t_path, intensities)
all_measured = get_images(t_path, intensities)

sorting_idx = get_sorting_indicies(inverse_idx, total_int)


# %%
coord_mask = get_validity_mask_for_coords(
    lim_0=(-4, -3),
    lim_1=(4, 6),
    scan_coords=scan_coords,
)

intensity_mask = get_validity_mask_for_intensities(
    lim=(1e3, 1e9),
    total_int=total_int,
)

valid_mask = coord_mask & intensity_mask

unique_pos_number = len(np.unique(scan_coords[valid_mask], axis=0))

unique_ints = all_measured[sorting_idx[valid_mask][:unique_pos_number]]

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


# def validate_dataloader_config(config: dict):
#     """
#     Validates the dataloader configuration dictionary.

#     Args:
#         config (dict): Configuration dictionary to validate.

#     Raises:
#         ValueError: If any required key is missing or has an invalid value.
#     """
#     required_keys = [
#         "file_path",
#         "scan_coord_paths",
#         "calculate_avg_image",
#         "avg_image_path",
#         "load_probe_image",
#         "probe_image_path",
#         "calculate_sortings",
#         "save_sortings",
#         "mask_path",
#         "precise_mask_path",
#     ]

#     for key in required_keys:
#         if key not in config:
#             raise ValueError(f"Missing required key: {key}")

#     if (
#         not isinstance(config["scan_coord_paths"], tuple)
#         or len(config["scan_coord_paths"]) != 2
#     ):
#         raise ValueError("scan_coord_paths must be a tuple of two strings.")


def load_dataset(loader_config: dict) -> dict[str, np.ndarray]:
    """
    Loads the dataset based on the provided loader configuration.

    Args:
        loader_config (dict): Configuration dictionary containing paths and parameters for loading the dataset.

    Returns:
        dict: Dictionary containing loaded dataset components.
    """
    # Validate the loader configuration
    # validate_dataloader_config(loader_config)

    # Load scan coordinates

    print(f"Loading data from config: {loader_config}")
    scan_coords = read_scan_coordinates_from_h5(
        loader_config["file_path"], loader_config["scan_coord_paths"]
    )

    print(f"Loaded scan coordinates with shape: {scan_coords.shape}")

    # Load total intensities and images

    all_measured = get_images(loader_config["file_path"], "images")
    total_int = np.nansum(all_measured, axis=(-1, -2))

    print(f"Loaded measured intensities with shape: {all_measured.shape}")

    # Get unique positions and sorting indices
    unique, first_idx, inverse_idx, counts = get_unique_positions(scan_coords)
    sorting_idx = get_sorting_indicies(inverse_idx, total_int)

    print(f"Found {len(unique)} unique scan positions.")

    # Create validity masks
    if "scan_coord_limits" in loader_config:
        coord_mask = get_validity_mask_for_coords(
            lim_0=loader_config["scan_coord_limits"]["0"],
            lim_1=loader_config["scan_coord_limits"]["1"],
            scan_coords=scan_coords,
        )
    else:
        coord_mask = np.ones(scan_coords.shape[0], dtype=bool)

    print(f"Created coordinate validity mask with {np.sum(coord_mask)} valid entries.")

    if "intensity_limits" in loader_config:
        intensity_mask = get_validity_mask_for_intensities(
            lim=loader_config["intensity_limits"], total_int=total_int
        )
        valid_mask = coord_mask & intensity_mask
        print(
            f"Created intensity validity mask with {np.sum(intensity_mask)} valid entries."
        )
        print(
            f"Created combined validity mask with {np.sum(valid_mask)} valid entries."
        )
    else:
        valid_mask = coord_mask

        print(
            f"Created combined validity mask with {np.sum(valid_mask)} valid entries."
        )

    # Filter unique positions and intensities based on validity mask
    unique_pos_number = len(np.unique(scan_coords[valid_mask], axis=0))
    unique_ints = all_measured[sorting_idx[valid_mask][:unique_pos_number]]

    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(np.nansum(all_measured[sorting_idx, ...], axis=(-1, -2)))

    # load mask
    mask = get_images(loader_config["file_path"], loader_config["mask_path"])
    if "precise_mask_path" in loader_config:
        precise_mask = get_images(
            loader_config["file_path"], loader_config["precise_mask_path"]
        )
        mask = mask * precise_mask  # Combine masks if precise mask is provided

    print(f"Loaded mask with shape: {mask.shape}")
    # load average image and probe if specified in the config

    if loader_config.get("calculate_avg_image", False):
        mean_image = np.nanmean(unique_ints, axis=0)
        print(f"Calculated average image with shape: {mean_image.shape}")
    else:
        if "avg_image_path" in loader_config:
            mean_image = get_images(
                loader_config["file_path"], loader_config["avg_image_path"]
            )
            print(f"Loaded average image with shape: {mean_image.shape}")
        else:
            raise ValueError(
                "avg_image_path must be provided if calculate_avg_image is False."
            )

    if loader_config.get("load_probe_image", False):
        if "probe_image_path" in loader_config:
            mean_probe = get_images(
                loader_config["file_path"], loader_config["probe_image_path"]
            )
            print(f"Loaded probe image with shape: {mean_probe.shape}")
        else:
            raise ValueError(
                "probe_image_path must be provided if load_probe_image is True."
            )

    return {
        "scan_coords": scan_coords[sorting_idx, :][valid_mask[sorting_idx], :],
        "total_ints": total_int[sorting_idx][valid_mask[sorting_idx]],
        "measured_intensities": all_measured[sorting_idx, ...][
            valid_mask[sorting_idx], ...
        ],
        "position_id": inverse_idx[valid_mask[sorting_idx]],
        "mean_image": mean_image,
        "mean_probe": mean_probe,
        "mask": mask,
    }


def load_exp_params_from_attributes(
    file_path: str, experimental_parameters: tuple
) -> dict:
    """
    Loads experimental parameters from the attributes of an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        experimental_parameters (tuple): Tuple of parameter names to load.

    Returns:
        dict: Dictionary containing the loaded experimental parameters.
    """
    with h5py.File(file_path, "r") as f:
        experimental_parameters = {
            param: f.attrs[param] for param in experimental_parameters
        }
        print("")
        print("Experimental parameters loaded from HDF5 file attributes:")
        for key, value in experimental_parameters.items():
            print(f"{key}: {value}")

        print("")
    return experimental_parameters


def get_reconstruction_resolution(
    detector_pixel_size_m, pixel_number, sample_detector_distance_m, wavelength_m
):
    """
    Calculate the reconstruction resolution based on the provided parameters.

    Parameters:
    - detector_pixel_size_m: Size of a pixel on the detector in meters.
    - pixel_number: Number of pixels on the detector.
    - sample_detector_distance_m: Distance from the sample to the detector in meters.
    - wavelength_m: Wavelength of the light used in meters.

    Returns:
    - resolution: The calculated reconstruction resolution.
    """
    # Calculate the angular resolution
    angular_resolution = wavelength_m / (detector_pixel_size_m * pixel_number)

    # Calculate the spatial resolution
    resolution = sample_detector_distance_m * angular_resolution

    return resolution


def process_scan_coordinates(
    scan_coords: np.ndarray,
    reconstruction_pix_size: float,
    scan_coords_unit: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Processes the scan coordinates by converting them to pixels, centering them, and splitting into integer and float parts.

    Args:
        scan_coords (np.ndarray): The scan coordinates in the unit specified by `scan_coords_unit`.
        reconstruction_pix_size (float): The pixel size for reconstruction in meters.
        scan_coords_unit (float): The unit of the scan coordinates (default is 1e-3 for millimeters).

    Returns:
        tuple: A tuple containing the integer and float parts of the processed scan coordinates.
    """
    # Convert to pixels
    scan_coords_pix = scan_coords * scan_coords_unit / reconstruction_pix_size

    # Center the coordinates
    scan_coords_centered = scan_coords_pix - np.mean(scan_coords_pix, axis=0)

    # Split into integer and float parts
    int_parts = np.floor(scan_coords_centered).astype(int)
    float_parts = scan_coords_centered - int_parts

    return int_parts, float_parts


################


if __name__ == "__main__":
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

    loaded = load_dataset(dataloader_config)
    print("")
    print("")

    for key, value in loaded.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"{key}: {value}")
    # %%
