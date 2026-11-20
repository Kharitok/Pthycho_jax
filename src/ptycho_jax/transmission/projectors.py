"""
Describes projection operators used during the reconstruction.

The idea is to perform projections after every gradient update.
Projection operators are defined as functions of (diff_params, non_diff_params)
that return the projected versions of them.

Type Conventions:
  - Projections operate on pytrees (dicts, arrays) representing model parameters.
  - Single-output projections: array → array.
  - Multi-parameter projections: (*arrays) → (*arrays) or (array, array) → (array, array).
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp

# projections


def create_frequency_space_mask_projector(mask: jnp.ndarray) -> Callable:
    """
    Creates a projector that applies a mask in frequency space.

    The mask is applied to the FFT of the input, then the result is transformed back.

    Parameters
    ----------
    mask : jnp.ndarray
        2D mask array, same shape as the last two dimensions of input arrays.

    Returns
    -------
    Callable
        JIT-compiled function: array → array with frequency mask applied.
    """

    def proj_func(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.ifft2(jnp.fft.fft2(x, axes=(-1, -2)) * mask, axes=(-1, -2))

    return jax.jit(proj_func, donate_argnames=["x"])


def create_real_space_mask_projector(mask: jnp.ndarray) -> Callable:
    """
    Creates a projector that applies a mask in real space (element-wise multiplication).

    Parameters
    ----------
    mask : jnp.ndarray
        2D mask array, same shape as the last two dimensions of input arrays.
        Typically binary (0 or 1) to indicate valid regions.

    Returns
    -------
    Callable
        JIT-compiled function: array → array with real-space mask applied.
    """

    def proj_func(x: jnp.ndarray) -> jnp.ndarray:
        return x * mask

    return jax.jit(proj_func, donate_argnames=["x"])


def create_limiter_scaling_complex(
    max_mod_val: float, min_mod_val: float = 0.0
) -> Callable:
    """
    Creates a projection that scales the modulus of a complex array to a specified range,
    preserving phase.

    The modulus is scaled linearly: mod' = (mod / mod_max) * (max - min) + min.

    Parameters
    ----------
    max_mod_val : float
        Maximum modulus value after projection.
    min_mod_val : float, optional
        Minimum modulus value after projection. Default: 0.

    Returns
    -------
    Callable
        JIT-compiled function: complex_array → scaled_complex_array.
    """

    def proj_func(x: jnp.ndarray) -> jnp.ndarray:
        module = jnp.abs(x)
        max_module = module.max()
        # Avoid division by zero
        max_module = jnp.where(max_module > 0, max_module, 1.0)
        scaled = (module / max_module) * (max_mod_val - min_mod_val) + min_mod_val
        return jnp.exp(1j * jnp.angle(x)) * scaled

    return jax.jit(proj_func, donate_argnames=["x"])


def create_modulus_clipper(
    max_mod_val: float = 1.0, min_mod_val: float = 0.0
) -> Callable:
    """
    Creates a projection that strictly clamps the modulus of a complex array
    between min and max values, preserving phase efficiently.
    """

    def proj_func(x: jnp.ndarray) -> jnp.ndarray:
        mod = jnp.abs(x)

        # 1. Pixel-wise clipping (leaves valid pixels completely untouched)
        mod_clipped = jnp.clip(mod, min=min_mod_val, max=max_mod_val)

        # 2. Safe division to prevent NaNs at exactly zero amplitude
        # If mod is 0, we substitute 1.0 for the denominator.
        # (Since x is 0, the numerator x will safely zero out the result anyway).
        safe_mod = jnp.where(mod > 0, mod, 1.0)

        # 3. Apply the scaling ratio directly to the complex number
        # No trig functions (angle/exp) required!
        scaling_ratio = mod_clipped / safe_mod

        # jax will broadcast the real ratio across the complex array
        return x * scaling_ratio

    return jax.jit(proj_func, donate_argnames=["x"])


def _remove_phase_gradient(x: jnp.ndarray) -> jnp.ndarray:
    """
    Remove phase gradients along axes 0 and 1 by subtracting row and column means.

    Useful for probe/sample wavefronts to center the average phase.
    """
    phase = jnp.angle(x)
    phase_centered = phase - phase.mean(axis=0)[None, :] - phase.mean(axis=1)[:, None]
    return phase_centered


def create_clipper_complex(
    max_mod_val: float, min_mod_val: float = 0.0, remove_phase_grad: bool = True
) -> Callable:
    """
    Creates a projection that clips the modulus of a complex array to a specified range.

    Optionally removes phase gradients (row and column mean phases) to center the wavefront.

    Parameters
    ----------
    max_mod_val : float
        Maximum modulus value after clipping.
    min_mod_val : float, optional
        Minimum modulus value after clipping. Default: 0.
    remove_phase_grad : bool, optional
        If True, subtract row/column-averaged phase to remove linear phase gradients.
        Default: True.

    Returns
    -------
    Callable
        JIT-compiled function: complex_array → clipped_complex_array.
    """

    def proj_func(x: jnp.ndarray) -> jnp.ndarray:
        module = jnp.abs(x)
        clipped_module = jnp.clip(module, min_mod_val, max_mod_val)

        if remove_phase_grad:
            phase = _remove_phase_gradient(x)
        else:
            phase = jnp.angle(x)

        return jnp.exp(1j * phase) * clipped_module

    return jax.jit(proj_func, donate_argnames=["x"])


def create_clipper_real(max_val: float, min_val: float) -> Callable:
    """
    Creates a projection that clips real-valued array elements to a specified range.

    Parameters
    ----------
    max_val : float
        Maximum allowable value.
    min_val : float
        Minimum allowable value.

    Returns
    -------
    Callable
        JIT-compiled function: array → clipped_array.
    """

    def proj_func(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, min_val, max_val)

    return jax.jit(proj_func, donate_argnames=["x"])


def create_value_centering_projection(axis: int = 0, bias: float = 0.0) -> Callable:
    """
    Creates a projection that centers array values along a specified axis around a bias.

    For a 2D array with axis=0, centers each column; with axis=1, centers each row.

    Parameters
    ----------
    axis : int, optional
        Axis along which to compute and subtract the mean. Default: 0 (column-wise).
    bias : float, optional
        Target bias value to add back after centering. Default: 0.

    Returns
    -------
    Callable
        JIT-compiled function: array → centered_array.
    """

    def proj_func(x: jnp.ndarray) -> jnp.ndarray:
        return x - x.mean(axis=axis, keepdims=True) + bias

    return jax.jit(proj_func, donate_argnames=["x"])


def create_shift_rounder(
    max_correction_magnitude: float, rounding_step: float
) -> Callable:
    """
    Creates a projection that rounds sub-pixel shift corrections to multiples of rounding_step.

    Keeps only the remainder (< rounding_step) in the differentiable correction part,
    and updates the non-differentiable pixel selector accordingly.

    This allows gradient descent to refine only fine corrections while coarser corrections
    stay fixed in the discrete position selector.

    Parameters
    ----------
    max_correction_magnitude : float
        Maximum differentiable correction magnitude (clamps via tanh mapping).
    rounding_step : float
        Quantization step size for rounding.

    Returns
    -------
    Callable
        JIT-compiled function: (fractional_correction, whole_positions) →
                               (updated_fractional, updated_whole).
    """

    def round_shifts(
        fractional: jnp.ndarray, whole: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Map fractional correction via tanh to enforce bounds
        fractional = jnp.tanh(fractional) * max_correction_magnitude
        # Round to nearest multiple of rounding_step
        correction = (
            jnp.sign(fractional)
            * rounding_step
            * (jnp.abs(fractional) // rounding_step)
        )
        # Update positions and remove quantized part from fractional
        whole = whole + correction
        fractional = fractional - correction
        # Map back to unconstrained space for optimization
        fractional = jnp.atanh(
            jnp.clip(fractional / max_correction_magnitude, -0.999, 0.999)
        )
        return jnp.float32(fractional), jnp.int16(whole)

    return jax.jit(round_shifts, donate_argnames=["whole", "fractional"])


def create_shift_limiter(
    max_allowed_shift: float, initial_positions: jnp.ndarray
) -> Callable:
    """
    Creates a projection that constrains scan position corrections to a maximum range
    from their initial values.

    Parameters
    ----------
    max_allowed_shift : float
        Maximum allowed absolute shift from initial position (in pixels).
    initial_positions : jnp.ndarray
        Initial scan positions, shape [n_positions, 2].

    Returns
    -------
    Callable
        JIT-compiled function: positions → clipped_positions.
    """

    def limit_shifts(whole_positions: jnp.ndarray) -> jnp.ndarray:
        # Relative shift from initial
        delta = whole_positions - initial_positions
        # Clip to allowed range
        delta = jnp.clip(delta, -max_allowed_shift, max_allowed_shift)
        # Return absolute positions
        return delta + initial_positions

    return jax.jit(limit_shifts, donate_argnames=["whole_positions"])


# Helper utilities


def _is_sequence_of_strings(obj: Any) -> bool:
    """Check if object is a sequence of strings (but not a string itself)."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def create_projection_applier(
    projection_func: Callable,
    diff_param_name: str | list[str] | None = None,
    non_diff_param_name: str | list[str] | None = None,
) -> Callable:
    """
    Creates a function that applies a projection to selected parameters.

    The projection_func receives arguments in order: *diff_params, *non_diff_params
    and returns outputs in the same order: *diff_outputs, *non_diff_outputs.

    Parameters
    ----------
    projection_func : Callable
        Projection function accepting the selected parameters and returning updated values
        in the same order: (diff1, diff2, ..., non_diff1, non_diff2, ...) →
                           (diff1', diff2', ..., non_diff1', non_diff2', ...).
    diff_param_name : str or list[str], optional
        Name(s) of differentiable parameters to project.
    non_diff_param_name : str or list[str], optional
        Name(s) of non-differentiable parameters to project.

    Returns
    -------
    Callable
        Function: (diff_params, non_diff_params) → (diff_params', non_diff_params').

    Raises
    ------
    ValueError
        If both parameter names are None.
    """
    if diff_param_name is None and non_diff_param_name is None:
        raise ValueError(
            "At least one of 'diff_param_name' or 'non_diff_param_name' must be specified"
        )

    diff_is_list = (
        _is_sequence_of_strings(diff_param_name) if diff_param_name else False
    )
    non_diff_is_list = (
        _is_sequence_of_strings(non_diff_param_name) if non_diff_param_name else False
    )

    if diff_param_name is None:
        # Only non-diff params
        if non_diff_is_list:
            non_diff_names = cast(list[str], non_diff_param_name)

            def apply_projection(diff_params: dict, non_diff_params: dict) -> tuple:
                inputs = [non_diff_params[k] for k in non_diff_names]
                outputs = projection_func(*inputs)
                updates = dict(zip(non_diff_names, outputs))
                return diff_params, {**non_diff_params, **updates}

        else:
            non_diff_name = cast(str, non_diff_param_name)

            def apply_projection(diff_params: dict, non_diff_params: dict) -> tuple:
                non_diff_params[non_diff_name] = projection_func(
                    non_diff_params[non_diff_name]
                )
                return diff_params, non_diff_params

    elif non_diff_param_name is None:
        # Only diff params
        if diff_is_list:
            diff_names = cast(list[str], diff_param_name)

            def apply_projection(diff_params: dict, non_diff_params: dict) -> tuple:
                inputs = [diff_params[k] for k in diff_names]
                outputs = projection_func(*inputs)
                updates = dict(zip(diff_names, outputs))
                return {**diff_params, **updates}, non_diff_params

        else:
            diff_name = cast(str, diff_param_name)

            def apply_projection(diff_params: dict, non_diff_params: dict) -> tuple:
                diff_params[diff_name] = projection_func(diff_params[diff_name])
                return diff_params, non_diff_params

    else:
        # Both diff and non-diff params
        if diff_is_list and non_diff_is_list:
            diff_names = cast(list[str], diff_param_name)
            non_diff_names = cast(list[str], non_diff_param_name)

            def apply_projection(diff_params: dict, non_diff_params: dict) -> tuple:
                diff_inputs = [diff_params[k] for k in diff_names]
                non_diff_inputs = [non_diff_params[k] for k in non_diff_names]
                outputs = projection_func(*diff_inputs, *non_diff_inputs)

                # Split outputs back
                num_diff = len(diff_names)
                diff_outputs = outputs[:num_diff]
                non_diff_outputs = outputs[num_diff:]

                diff_updates = dict(zip(diff_names, diff_outputs))
                non_diff_updates = dict(zip(non_diff_names, non_diff_outputs))
                return {**diff_params, **diff_updates}, {
                    **non_diff_params,
                    **non_diff_updates,
                }

        elif diff_is_list:
            diff_names = cast(list[str], diff_param_name)
            non_diff_name = cast(str, non_diff_param_name)

            def apply_projection(diff_params: dict, non_diff_params: dict) -> tuple:
                diff_inputs = [diff_params[k] for k in diff_names]
                outputs = projection_func(*diff_inputs, non_diff_params[non_diff_name])

                diff_outputs = outputs[:-1]
                non_diff_output = outputs[-1]

                diff_updates = dict(zip(diff_names, diff_outputs))
                return {**diff_params, **diff_updates}, {
                    **non_diff_params,
                    non_diff_name: non_diff_output,
                }

        elif non_diff_is_list:
            diff_name = cast(str, diff_param_name)
            non_diff_names = cast(list[str], non_diff_param_name)

            def apply_projection(diff_params: dict, non_diff_params: dict) -> tuple:
                non_diff_inputs = [non_diff_params[k] for k in non_diff_names]
                outputs = projection_func(diff_params[diff_name], *non_diff_inputs)

                diff_output = outputs[0]
                non_diff_outputs = outputs[1:]

                non_diff_updates = dict(zip(non_diff_names, non_diff_outputs))
                return {**diff_params, diff_name: diff_output}, {
                    **non_diff_params,
                    **non_diff_updates,
                }

        else:
            diff_name = cast(str, diff_param_name)
            non_diff_name = cast(str, non_diff_param_name)

            def apply_projection(diff_params: dict, non_diff_params: dict) -> tuple:
                diff_out, non_diff_out = projection_func(
                    diff_params[diff_name],
                    non_diff_params[non_diff_name],
                )
                diff_params[diff_name] = diff_out
                non_diff_params[non_diff_name] = non_diff_out
                return diff_params, non_diff_params

    return apply_projection


def merge_multiple_projections(
    proj_list: list[Callable],
) -> Callable:
    """
    Merges multiple projection functions into a single projection.

    Projections are applied sequentially in order. Each projection function should
    have the signature: (diff_params, non_diff_params) → (diff_params', non_diff_params').

    Parameters
    ----------
    proj_list : list[Callable]
        List of projection functions to apply in order.

    Returns
    -------
    Callable
        JIT-compiled function that applies all projections sequentially.

    Example
    -------
    >>> proj1 = create_projection_applier(create_clipper_real(1.0, 0.0), diff_param_name="x")
    >>> proj2 = create_projection_applier(create_value_centering_projection(), diff_param_name="x")
    >>> combined = merge_multiple_projections([proj1, proj2])
    >>> diff_params, non_diff_params = combined(diff_params, non_diff_params)
    """

    def apply_projections(
        differentiable_params: dict, non_diff_params: dict
    ) -> tuple[dict, dict]:
        for proj in proj_list:
            differentiable_params, non_diff_params = proj(
                differentiable_params, non_diff_params
            )
        return differentiable_params, non_diff_params

    return jax.jit(
        apply_projections,
        donate_argnames=["differentiable_params", "non_diff_params"],
    )
