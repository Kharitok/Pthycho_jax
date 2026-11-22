"""Array-only pytrees passed explicitly to compiled reconstruction functions."""

from typing import NamedTuple
import jax


class DMState(NamedTuple):
    """Exit waves (J,K,Y,X), probe (K,Y,X), object (H,W), amplitudes (J,K), scalar it."""

    psi: jax.Array
    probe: jax.Array
    obj: jax.Array
    a: jax.Array
    it: jax.Array


class Data(NamedTuple):
    """Measured amplitudes/mask (J,Y,X) and valid photon counts (J,)."""

    sqrt_I: jax.Array
    mask: jax.Array
    counts: jax.Array
