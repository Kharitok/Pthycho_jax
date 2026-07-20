"""
Describes forward model for transmission ptychography. The model is composed from
Sample_selector consstructiong and scanning the sample
Probe_selector constructing a probe
Proparagator performing the propagation to the detector plane
Detector converting wavefields into the measured intensities and taking into account possible nonlinear responce and detector atefacts

The model first implemented as a point model - description for the single intensity and then transformed into batch model using jax vmap
"""

# %%
from typing import Callable, Dict, Tuple, TypedDict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int32

### for typechecking


class OptimizableParams(TypedDict):
    # Enforce axis sizes and names natively
    probe_modes: Complex[Array, "n_modes det_y det_x"]
    modal_weights: Complex[Array, "num_pos n_modes m_probes"]
    sample: Complex[Array, "sample_y sample_x"]


class FixedParams(TypedDict):
    # You can add shapes, floats, ints, or general Any for fixed params
    wavelength: float
    pixel_size: float
    # ... other fixed param


### propagators
def fourier_propagator(
    exit_wave: Complex[Array, "n_modes det_y det_x"],
) -> Complex[Array, "n_modes det_y det_x"]:
    """
    Propagates the exit wave to the detector plane using Fourier transform.

    Args:
        exit_wave: The exit wavefield at the sample plane.

    Returns:
        The propagated wavefield at the detector plane.
    """
    return jnp.fft.fft2(exit_wave, norm="ortho", axes=(-2, -1))


def fresnel_propagator(
    exit_wave: Complex[Array, "n_modes det_y det_x"],
    wavelength: float,
    pixel_size: float,
    propagation_distance: float,
) -> Complex[Array, "n_modes det_y det_x"]:
    """
    Propagates the exit wave to the detector plane using Fresnel propagation.

    Args:
        exit_wave: The exit wavefield at the sample plane.
        wavelength: Wavelength of the light.
        pixel_size: Size of a pixel in the detector.
        propagation_distance: Distance from the sample to the detector.

    Returns:
        The propagated wavefield at the detector plane.
    """
    ny, nx = exit_wave.shape[-2:]
    k = 2 * jnp.pi / wavelength
    fx = jnp.fft.fftfreq(nx, d=pixel_size)
    fy = jnp.fft.fftfreq(ny, d=pixel_size)
    FX, FY = jnp.meshgrid(fx, fy)
    H = jnp.exp(-1j * (jnp.pi * wavelength * propagation_distance) * (FX**2 + FY**2))
    return jnp.fft.ifft2(jnp.fft.fft2(exit_wave) * H)


# %%


def point_model_ptycho_transmission(
    probe_constructor: Callable,
    sample_constructor: Callable,
    propagator: Callable,
    detector: Callable,
) -> Callable:
    """
    Forward model for transmission ptychography. The model is composed from
    Sample_selector constructiong and scanning the sample
    Probe_selector constructing a probe
    Proparagator performing the propagation to the detector plane
    Detector converting wavefields into the measured intensities and taking into account possible nonlinear responce and detector atefacts

    The model first implemented as a point model - description for the single intensity and then transformed into batch model using jax vmap

    Args:
        probe_constructor: function that constructs a probe
        sample_constructor: function that constructs a sample
        propagator: function that propagates the wavefield to the detector plane
        detector: function that converts wavefields into measured intensities

    Returns:
        A function that takes in the parameters for the probe, sample, and detector, and returns the measured intensities.
    """

    def point_model(
        optimizable_params: Dict, fixed_params: Dict, pos_index: Int32
    ) -> Float[Array, "det_y det_x"]:
        probe = probe_constructor(optimizable_params, fixed_params, pos_index)
        sample = sample_constructor(optimizable_params, fixed_params, pos_index)
        exit_wave = probe * sample[None, ...]
        propagated_wave = propagator(exit_wave)
        intensity = detector(
            propagated_wave, optimizable_params, fixed_params, pos_index
        )
        return intensity

    return point_model


def construct_point_model_ptycho_transmission(
    model_parameters: Dict,
) -> Dict[str, Callable]:
    """
    Constructs a point model for transmission ptychography based on the provided model parameters.
    """

    probe_type = model_parameters.get("probe_type", "static")
    if probe_type not in ["static", "fluctuating"]:
        raise ValueError(
            f"Invalid probe_type: {probe_type}. Must be 'static' or 'fluctuating'."
        )
    sample_type = model_parameters.get("sample_type", "complex")
    if sample_type not in [
        "complex",
        "refractive",
        "exp_refractive",
        "tanh_refractive",
    ]:
        raise ValueError(
            f"Invalid sample_type: {sample_type}. Must be 'complex', 'refractive', 'exp_refractive', or 'tanh_refractive'."
        )
    binning_factor = model_parameters.get("binning_factor", (1, 1))
    shift_margin = model_parameters.get("shift_margin", 1)
    sample_selector_type = model_parameters.get("sample_selector_type", "slicing")
    if sample_selector_type not in ["slicing", "correcting"]:
        raise ValueError(
            f"Invalid sample_selector_type: {sample_selector_type}. Must be 'slicing' or 'correcting'."
        )
    propagator_type = model_parameters.get("propagator_type", "Fourrier")
    if propagator_type not in ["Fourrier", "Fresnel"]:
        raise ValueError(
            f"Invalid propagator_type: {propagator_type}. Must be 'Fourrier' or 'Fresnel'."
        )

    if propagator_type == "Fourrier":
        propagator = fourier_propagator
    elif propagator_type == "Fresnel":
        propagator = fresnel_propagator

    if probe_type == "static":
        probe_constructor = static_probe
    elif probe_type == "fluctuating":
        probe_constructor = fluctuating_probe

    if sample_type == "complex":
        sample_scaler = identity_sample_scaler
    elif sample_type == "refractive":
        sample_scaler = refractive_sample_scaler
    elif sample_type == "exp_refractive":
        sample_scaler = exp_refractive_sample_scaler
    elif sample_type == "tanh_refractive":
        sample_scaler = tanh_refractive_sample_scaler

    if sample_selector_type == "slicing":
        sample_selector = prepare_slicing_sample_selector(
            model_parameters["probe_shape"], crop_from_side=shift_margin
        )
    elif sample_selector_type == "correcting":
        sample_selector = prepare_correcting_sample_selector(
            model_parameters["probe_shape"], crop_from_side=shift_margin
        )

    sample_constructor = prepare_sample_constructor(
        sample_selector=sample_selector, sample_scaler=sample_scaler
    )

    detector = prepare_detector(
        intensifier=intensifier,
        binner=prepare_detector_binning(binning_factor),
        non_linearity=identity_non_linearity,
    )
    point_model = point_model_ptycho_transmission(
        probe_constructor=probe_constructor,
        sample_constructor=sample_constructor,
        propagator=propagator,
        detector=detector,
    )
    return {
        "point_model": point_model,
        "propagator": propagator,
        "detector": detector,
        "probe_constructor": probe_constructor,
        "sample_constructor": sample_constructor,
    }


### Probe models ###
def static_probe(
    optimizable_params: Dict, fixed_params: Dict, pos_index: Int32
) -> Float[Array, "n_modes det_y det_x"]:
    """
    Constructs a static probe.

    Args:
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.

    Returns:
        A static probe as a JAX array.
    """
    return optimizable_params["probe_modes"]


def fluctuating_probe(
    optimizable_params: Dict, fixed_params: Dict, pos_index: Int32
) -> Float[Array, "m_modes det_y det_x"]:
    """
    Constructs a fluctuating probe. It is constructed from N optimizable probe modes into M incoherent probes that will be used for the propagation with NxM weight matrix.
    This mean that the partially coherent but static probe is diagonal in the weights space while fluctuating and partially coherent has a full matrix
    Args:
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.

    Returns:
        A fluctuating probe as a JAX array.
    """
    probe_modes = optimizable_params[
        "probe_modes"
    ]  # [N, detector_shape, detector_shape]
    modal_weights = optimizable_params["modal_weights"][pos_index]  # [N, M]
    return jnp.einsum("nxy,nm->mxy", probe_modes, modal_weights)


### Sample models ###
def prepare_sample_constructor(
    sample_selector: Callable, sample_scaler: Callable
) -> Callable:
    """
    Prepares a sample constructor function that selects and scans the sample.

    Args:
        sample_selector: Function that selects the sample patch and performs position scanning.
        sample_scaler: Function that scales the sample and converts it to the complex transfer function in case of the refractive sample.
    Returns:
        A function that constructs the sample given optimizable and fixed parameters, and position index.
    """

    def sample_constructor(
        optimizable_params: Dict, fixed_params: Dict, pos_index: Int32
    ) -> Complex[Array, "sample_y sample_x"]:
        sample = sample_scaler(optimizable_params, fixed_params, pos_index)
        sample_patch = sample_selector(
            sample, optimizable_params, fixed_params, pos_index
        )
        return sample_patch

    return sample_constructor


def extract_patch(
    object_arr: Array, corner_pixel: Tuple[int, int], patch_shape: Tuple[int, int]
):
    patch = jax.lax.dynamic_slice(object_arr, corner_pixel, patch_shape)
    return patch


def fourier_shift(patch, shift):
    dy, dx = shift
    ny, nx = patch.shape[-2:]
    fy = jnp.fft.fftfreq(ny)[:, None]
    fx = jnp.fft.fftfreq(nx)[None, :]
    ramp = jnp.exp(-2j * jnp.pi * (fy * dy + fx * dx))
    return jnp.fft.ifft2(jnp.fft.fft2(patch) * ramp)


def crop_patch(patch, crop_from_side):
    return patch[crop_from_side:-crop_from_side, crop_from_side:-crop_from_side]


def prepare_slicing_sample_selector(
    probe_shape: Tuple[int, int], crop_from_side: int = 0
) -> Callable:
    """
    Prepares a sample selector function that selects a patch of the sample based on the probe shape and position index.

    Args:
        probe_shape: Shape of the probe (height, width).
        crop_from_side: Number of pixels to crop from each side of the selected patch.
    Returns:
        A function that selects a patch of the sample given the sample array and position index.
    """
    if crop_from_side == 0:

        def sample_selector(
            sample: Complex[Array, "sample_y sample_x"],
            optimizable_params: Dict,
            fixed_params: Dict,
            pos_index: Int32,
        ) -> Complex[Array, "patch_y patch_x"]:
            # pos = fixed_params["sample_positions"][pos_index, ...]
            corner_pixel = fixed_params["sample_positions"][pos_index, ...]
            patch = extract_patch(
                sample,
                corner_pixel,
                (
                    int(probe_shape[0] + 2 * crop_from_side),
                    int(probe_shape[1] + 2 * crop_from_side),
                ),
            )

            return patch

    else:

        def sample_selector(
            sample: Complex[Array, "sample_y sample_x"],
            optimizable_params: Dict,
            fixed_params: Dict,
            pos_index: Int32,
        ) -> Complex[Array, "patch_y patch_x"]:
            # pos = fixed_params["sample_positions"][pos_index, ...]
            corner_pixel = fixed_params["sample_positions"][pos_index, ...]
            patch = extract_patch(
                sample,
                corner_pixel,
                (
                    int(probe_shape[0] + 2 * crop_from_side),
                    int(probe_shape[1] + 2 * crop_from_side),
                ),
            )
            patch = crop_patch(patch, crop_from_side)
            return patch

    return sample_selector


def prepare_correcting_sample_selector(
    probe_shape: Tuple[int, int], crop_from_side: int = 1
) -> Callable:
    """
    Prepares a sample selector function that selects a patch of the sample based on the probe shape and position index, and applies a Fourier shift correction.

    Args:
        probe_shape: Shape of the probe (height, width).
        crop_from_side: Number of pixels to crop from each side of the selected patch.
    Returns:
        A function that selects a patch of the sample given the sample array and position index, and applies a Fourier shift correction.
    """
    if crop_from_side < 1:
        raise ValueError(
            "crop_from_side must be at least 1 for the correcting sample selector."
        )

    def sample_selector(
        sample: Complex[Array, "sample_y sample_x"],
        optimizable_params: Dict,
        fixed_params: Dict,
        pos_index: Int32,
    ) -> Complex[Array, "patch_y patch_x"]:
        # pos = fixed_params["sample_positions"][pos_index, ...]
        corner_pixel = fixed_params["sample_positions"][pos_index, ...]
        patch = extract_patch(
            sample,
            corner_pixel,
            (
                int(probe_shape[0] + 2 * crop_from_side),
                int(probe_shape[1] + 2 * crop_from_side),
            ),
        )
        shifts = optimizable_params["scan_mistakes"][pos_index, ...]
        patch = fourier_shift(patch, shifts)
        patch = crop_patch(patch, crop_from_side)
        # Apply Fourier shift correction

        return patch

    return sample_selector


# sample scalers
def identity_sample_scaler(
    optimizable_params: Dict,
    fixed_params: Dict,
    pos_index: Int32,
) -> Complex[Array, "sample_y sample_x"]:
    """
    Identity function for sample scaling. Returns the sample patch as is.

    Args:
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.

    Returns:
        The scaled sample patch as a complex array.
    """
    return optimizable_params["sample"]


def refractive_sample_scaler(
    sample_patch: Complex[Array, "sample_y sample_x"],
    optimizable_params: Dict,
    fixed_params: Dict,
    pos_index: Int32,
) -> Complex[Array, "sample_y sample_x"]:
    """
    Scales the sample patch and converts it to the complex transfer function for refractive samples.

    Args:
        sample_patch: The selected sample patch.
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.

    Returns:
        The scaled sample patch as a complex array.
    """

    raise NotImplementedError("Refractive sample scaler is not implemented yet.")


def exp_refractive_sample_scaler(
    sample_patch: Complex[Array, "sample_y sample_x"],
    optimizable_params: Dict,
    fixed_params: Dict,
    pos_index: Int32,
) -> Complex[Array, "sample_y sample_x"]:
    """
    Scales the sample patch and converts it to the complex transfer function for refractive samples using the exponential model.

    Args:
        sample_patch: The selected sample patch.
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.

    returns:
        The scaled sample patch as a complex array.
    """
    raise NotImplementedError(
        "Exponential refractive sample scaler is not implemented yet."
    )


def tanh_refractive_sample_scaler(
    sample_patch: Complex[Array, "sample_y sample_x"],
    optimizable_params: Dict,
    fixed_params: Dict,
    pos_index: Int32,
) -> Complex[Array, "sample_y sample_x"]:
    """
    Scales the sample patch and converts it to the complex transfer function for refractive samples using the hyperbolic tangent model.

    Args:
        sample_patch: The selected sample patch.
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.

    Returns:
        The scaled sample patch as a complex array.
    """
    raise NotImplementedError(
        "Hyperbolic tangent refractive sample scaler is not implemented yet."
    )


# Detector models


def prepare_detector(
    intensifier: Callable,
    binner: Callable,
    non_linearity: Callable,
) -> Callable:
    """
    Prepares a detector function that converts wavefields into measured intensities.

    Args:
        intensifier: Function that applies an intensifier to the propagated wavefield.
        binner: Function that applies binning to the propagated wavefield.
        non_linearity: Function that applies a non-linear response to the propagated wavefield.

    Returns:
        A function that takes in the propagated wavefield and returns the measured intensities.
    """

    def detector(
        propagated_wave: Complex[Array, "det_y det_x"],
        optimizable_params: Dict,
        fixed_params: Dict,
        pos_index: Int32,
    ) -> Float[Array, "det_y det_x"]:
        intensity = intensifier(
            propagated_wave, optimizable_params, fixed_params, pos_index
        )
        intensity = binner(intensity, optimizable_params, fixed_params, pos_index)
        intensity = non_linearity(
            intensity, optimizable_params, fixed_params, pos_index
        )
        return intensity

    return detector


def intensifier(
    propagated_wave: Complex[Array, "n_modes det_y det_x"],
    optimizable_params: Dict,
    fixed_params: Dict,
    pos_index: Int32,
) -> Float[Array, "det_y det_x"]:
    """
    Identity function for intensifier. Returns the propagated wavefield as is.

    Args:
        propagated_wave: The propagated wavefield.
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.
    Returns:
        The propagated wavefield as a complex array.
    """
    return (jnp.abs(propagated_wave) ** 2).sum(axis=0)  # Sum over modes if present


def identity_non_linearity(
    intensity: Float[Array, "det_y det_x"],
    optimizable_params: Dict,
    fixed_params: Dict,
    pos_index: Int32,
) -> Float[Array, "det_y det_x"]:
    """
    Identity function for non-linearity. Returns the intensity as is.

    Args:
        intensity: The measured intensity.
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.
    Returns:
        The measured intensity as a float array.
    """
    return intensity


def identity_binning(
    intensity: Float[Array, "det_y det_x"],
    optimizable_params: Dict,
    fixed_params: Dict,
    pos_index: Int32,
) -> Float[Array, "det_y det_x"]:
    """
    Identity function for binning. Returns the intensity as is.

    Args:
        intensity: The measured intensity.
        optimizable_params: Dictionary containing optimizable parameters.
        fixed_params: Dictionary containing fixed parameters.
        pos_index: Index of the position in the scan.
    Returns:
        The measured intensity as a float array.
    """
    return intensity


def prepare_detector_binning(binning_factor: Tuple[int, int]) -> Callable:
    """
    Prepares a binning function that applies binning to the propagated wavefield.

    Args:
        binning_factor: Tuple containing the binning factors for the y and x dimensions.

    Returns:
        A function that takes in the propagated wavefield and returns the binned intensities.
    """

    def binner(
        intensity: Float[Array, "det_y det_x"],
        optimizable_params: Dict,
        fixed_params: Dict,
        pos_index: Int32,
    ) -> Float[Array, "binned_det_y binned_det_x"]:
        binned_intensity = jax.lax.reduce_window(
            intensity,
            0.0,
            jax.lax.add,
            window_dimensions=binning_factor,
            window_strides=binning_factor,
            padding="VALID",
        )
        return binned_intensity

    return binner
