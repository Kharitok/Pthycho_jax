"""Reference-compatible FFT, amplitude weights and complex adjoints."""

import jax
import jax.numpy as jnp
import numpy as np


def fft2c(x):
    """Orthonormal unshifted FFT of (...,Y,X); data are already ifftshifted."""
    return jnp.fft.fft2(x, axes=(-2, -1), norm="ortho")


def ifft2c(x):
    """Inverse orthonormal FFT of (...,Y,X), without a spatial shift."""
    return jnp.fft.ifft2(x, axes=(-2, -1), norm="ortho")


def a_to_weights_param(a):
    """DM real amplitude (J,K) -> AD complex diagonal amplitude matrix (J,K,K)."""
    return jax.vmap(jnp.diag)(a)


def weights_to_a(weights):
    """AD diagonal nonnegative amplitude matrices (J,K,K) -> DM (J,K).

    General complex/coherent mixing matrices cannot be represented by diagonal
    nonnegative DM weights and are deliberately rejected outside compiled code.
    """
    values = np.asarray(weights)
    diagonal = np.diagonal(values, axis1=-2, axis2=-1)
    if (
        values.ndim != 3
        or values.shape[1] != values.shape[2]
        or np.any(
            np.abs(values - np.eye(values.shape[1])[None] * diagonal[:, None, :])
            > 1e-12
        )
        or np.any(np.abs(diagonal.imag) > 1e-12)
        or np.any(diagonal.real < 0)
    ):
        raise ValueError("DM requires diagonal nonnegative amplitude weights.")
    return jnp.asarray(diagonal.real)


def split_positions(positions):
    """Real positions (J,2) -> integer floor (J,2), fractional remainder (J,2)."""
    values = np.asarray(positions)
    integer = np.floor(values).astype(np.int32)
    return integer, values - integer


def hermitian_adjoint(function, example):
    """Hermitian adjoint of a complex-linear map; shape inferred from example."""
    transpose = jax.linear_transpose(function, example)
    return lambda y: jnp.conj(transpose(jnp.conj(y))[0])
