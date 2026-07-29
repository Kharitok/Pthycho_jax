"""
Loss and regularization functions for complex optimization.
"""

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32


#
def gauss_loss(approx_intensity, root_of_measured_intensity, mask):
    """
    Computes a loss between measured and approximated intensity using
    a Gaussian likelihood approximation for the Poisson noise.

    Parameters
    ----------
    approx_intensity : array
        Approximated intensity from the forward model.
    root_of_measured_intensity : array
        Square root of the measured intensity.
    index : int
        Index of the measurement in the batch.
    mask : array
        Mask to be applied to the approximated intensity.

    Returns
    -------
    loss : float
        Mean squared error between the square root of the measured intensity
        and the approximated intensity, optionally masked.
    """
    return jnp.mean(
        (jnp.sqrt(approx_intensity + 1e-20) - root_of_measured_intensity) ** 2,
        where=mask,
    )


def poisson_loss(approx_intensity, measured_intensity, mask):
    """
    Computes a loss between measured and approximated intensity using
    a Poisson likelihood.

    Parameters
    ----------
    approx_intensity : array
        Approximated intensity from the forward model.
    measured_intensity : array
        Measured intensity.
    mask : array
        Mask to be applied to the approximated intensity.

    Returns
    -------
    loss : float
        Poisson negative log-likelihood between the measured and approximated intensity,
        optionally masked.
    """
    return jnp.mean(
        approx_intensity - measured_intensity * jnp.log(approx_intensity + 1e-20),
        where=mask,
    )


def join_loss_and_forward(loss_func: Callable, forward_func: Callable) -> Callable:
    """
    Combines a loss function and a forward model function into a single callable.

    Parameters
    ----------
    loss_func : Callable
        A loss function that takes approximated intensity, measured intensity, and mask as inputs.
    forward_func : Callable
        A forward model function that takes parameters and returns approximated intensity.

    Returns
    -------
    combined_func : Callable
        A function that takes parameters, measured intensity, and mask as inputs,
        computes the approximated intensity using the forward model, and then computes the loss.
    """

    def combined_func(
        optimizable_params: Dict,
        fixed_params: Dict,
        pos_index: Int32,
        measured_intensity: Array,
        mask: Array,
    ) -> Float:
        approx_intensity = forward_func(optimizable_params, fixed_params, pos_index)
        return loss_func(approx_intensity, measured_intensity, mask)

    return combined_func


def join_loss_and_forward_batched(
    loss_func: Callable, forward_func: Callable
) -> Callable:
    """
    Builds a batched loss that averages per-position losses over the batch.

    The returned function expects `pos_index` and `measured_intensity` to share the
    leading batch dimension, then computes the mean of the per-position losses.
    """

    single_loss = join_loss_and_forward(loss_func, forward_func)

    def combined_func(
        optimizable_params: Dict,
        fixed_params: Dict,
        pos_index: Array,
        measured_intensity: Array,
        mask: Array,
    ) -> Float:
        losses = jax.vmap(single_loss, in_axes=(None, None, 0, 0, None))(
            optimizable_params,
            fixed_params,
            pos_index,
            measured_intensity,
            mask,
        )
        return jnp.mean(losses)

    return combined_func


### Regularization
# THis describes factories for the regularization functions
# Multiple regularizers should be possible to combine into one composite one.
# THe produced function should be then attached to the loss function as a regularization term,
#
# with the weights stored in the fixed params structure. This is implemented in join_loss_and_reg.
#


def create_tv_reg(p, q, epsilon=1e-6, conv_func=None):
    """

    [Wu, Longlong, et al. "Dose-efficient automatic differentiation for ptychographic reconstruction." Optica 11.6 (2024): 821-830.]

    Creates a total variation (TV) regularization function.

    The returned function computes the TV regularization of an input array `x`
    using the specified parameters `p` and `q`. The TV regularization is a
    measure of the smoothness of `x`, penalizing differences along the last
    two axes.

    Parameters
    ----------
    p : float
        The power to which the absolute differences are raised.
    q : float
        The power to which the mean of the summed differences is raised.
    epsilon : float, optional
        A small constant added for numerical stability in the smooth approximation of the absolute value.

    Returns
    -------
    tv_reg : callable
        A function that takes an input array `x` and returns the computed TV
        regularization value as a float.
    """
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if conv_func is None:
        conv_func = lambda x: x

    def tv_reg(x):
        x = conv_func(x)
        # Use smooth approximation of abs(x) = sqrt(x^2 + eps^2)
        diff_x = jnp.diff(x, n=1, axis=-1)[..., :-1, :]
        diff_y = jnp.diff(x, n=1, axis=-2)[..., :, :-1]

        smooth_abs_x = jnp.sqrt(jnp.abs(diff_x) ** 2 + epsilon**2)
        smooth_abs_y = jnp.sqrt(jnp.abs(diff_y) ** 2 + epsilon**2)

        return jnp.mean((smooth_abs_x**p + smooth_abs_y**p + 1e-20) ** (q / p))

    return tv_reg


def create_regularizer(diff_params_name, reg_weight_name: str, reg_func):
    """
    Creates a regularization function that computes the regularization term
    based on the specified parameters and regularization function.

    Parameters
    ----------
    diff_params_name : str
        The name of the differentiable parameters in the parameter dictionary.
    reg_weight_name : str
        The name of the regularization weight in the parameter dictionary.
    reg_func : callable
        A function that takes an input array and returns a regularization value.

    Returns
    -------
    regularizer : callable
        A function that takes a parameter dictionary and returns the computed
        regularization value as a float.
    """

    def regularizer(params, fixed_params):
        diff_parameter = params[diff_params_name]
        reg_weight = fixed_params[reg_weight_name]
        return reg_weight * reg_func(diff_parameter)

    return regularizer


def combine_regularizers(reg_list: list) -> Callable:
    """
    Combines multiple regularization functions into a single regularization function.

    Parameters
    ----------
    reg_list : list of callable
        A list of regularization functions to be combined.

    Returns
    -------
    combined_regularizer : callable
        A function that takes a parameter dictionary and returns the sum of the
        computed regularization values from all provided regularizers.
    """

    def combined_regularizer(params, fixed_params):
        total_reg = 0.0
        for reg in reg_list:
            total_reg += reg(params, fixed_params)
        return total_reg

    return combined_regularizer


def join_loss_and_reg(loss_and_forward_func: Callable, reg_func: Callable) -> Callable:
    """
    Combines a loss function and a regularization function into a single callable.

    Parameters
    ----------
    loss_func : Callable
        A loss function that takes parameters, fixed parameters, position index,
        measured intensity, and mask as inputs.
    reg_func : Callable
        A regularization function that takes parameters and fixed parameters as inputs.

    Returns
    -------
    combined_func : Callable
        A function that takes parameters, fixed parameters, position index,
        measured intensity, and mask as inputs, computes the loss and regularization,
        and returns their sum.
    """

    def combined_func(
        optimizable_params: Dict,
        fixed_params: Dict,
        pos_index: Array,
        measured_intensity: Array,
        mask: Array,
    ) -> Float:
        loss = loss_and_forward_func(
            optimizable_params, fixed_params, pos_index, measured_intensity, mask
        )
        reg = reg_func(optimizable_params, fixed_params)
        return loss + reg

    return combined_func
