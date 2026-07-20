"""
Loss and regularization functions for complex optimization.
"""

import jax
import jax.numpy as jnp


#
def l_gauss_loss(approx_intensity, root_of_measured_intensity, mask):
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
