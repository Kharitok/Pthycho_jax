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


def join_loss_and_forward_batched(loss_func: Callable, forward_func: Callable) -> Callable:
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
