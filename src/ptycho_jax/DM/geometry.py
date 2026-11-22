"""Small fixed-position geometry pytree; no persistent per-pixel indices."""

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["n", "delta"],
    meta_fields=["canvas_shape", "frame_shape", "margin", "integer_only"],
)
@dataclass(frozen=True)
class Geometry:
    n: jax.Array
    delta: jax.Array
    canvas_shape: tuple
    frame_shape: tuple
    margin: int
    integer_only: bool


def make_geometry(n, delta, canvas_shape, frame_shape, margin=10):
    """Validate fixed crop origins/fractions (J,2), canvas (H,W), frame (Y,X)."""
    n, delta = np.asarray(n), np.asarray(delta)
    if n.ndim != 2 or n.shape[1] != 2 or delta.shape != n.shape or len(n) == 0:
        raise ValueError("Expected nonempty (shots,2) origins and fractions.")
    if not np.isfinite(delta).all() or not np.all(n == np.floor(n)) or margin < 1:
        raise ValueError("Invalid integer crop origin, fraction or margin.")
    canvas_shape, frame_shape = (
        tuple(map(int, canvas_shape)),
        tuple(map(int, frame_shape)),
    )
    if np.any(n < 0) or np.any(n + np.array(frame_shape) + 2 * margin > canvas_shape):
        raise ValueError("Every enlarged object crop must fit the canvas.")
    return Geometry(
        jnp.asarray(n, dtype=jnp.int32),
        jnp.asarray(delta),
        canvas_shape,
        frame_shape,
        int(margin),
        bool(np.all(delta == 0)),
    )
