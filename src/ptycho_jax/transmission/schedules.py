"""Switch loss penalties and post-update constraints without restarting Adam."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

STEP_KEY = "_constraint_step"


def schedule_mask(specs: Sequence[str] | None, n_steps: int) -> np.ndarray:
    """Return active updates for zero-based indices or START:STOP[:EVERY].

    STOP is exclusive; omitted START/STOP mean zero/n_steps. Multiple specs
    form a union. None or 'all' enables every update; 'none' enables no updates.
    Out-of-range indices and empty intervals are rejected to expose typos.
    """
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if isinstance(specs, str):
        specs = (specs,)
    if specs is None or tuple(specs) == ("all",):
        return np.ones(n_steps, dtype=bool)
    if tuple(specs) == ("none",):
        return np.zeros(n_steps, dtype=bool)
    if not specs:
        raise ValueError("Schedule must contain a range, index, 'all' or 'none'.")
    mask = np.zeros(n_steps, dtype=bool)
    for spec in specs:
        try:
            fields = spec.strip().split(":")
            if len(fields) == 1:
                start = int(fields[0])
                stop, stride = start + 1, 1
            elif len(fields) in (2, 3):
                start = int(fields[0]) if fields[0] else 0
                stop = int(fields[1]) if fields[1] else n_steps
                stride = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            else:
                raise ValueError
            if not 0 <= start < stop <= n_steps or stride < 1:
                raise ValueError
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid schedule {spec!r}: use INDEX or START:STOP[:EVERY] "
                f"within [0, {n_steps}), or 'all'/'none' alone."
            ) from error
        mask[start:stop:stride] = True
    return mask


def schedule_constraints(
    regularizer: Callable,
    projection: Callable,
    regularization_mask: np.ndarray,
    projection_mask: np.ndarray,
) -> tuple[Callable, Callable]:
    """Gate callbacks using a counter in fixed parameters, advanced after updates.

    Initialize fixed[STEP_KEY] to int32(0) and pass both returned callbacks to
    one optimization-loop call. Every batch in a full pass sees the same step.
    The projection wrapper advances the counter even on unconstrained updates.
    Pure JAX callbacks must preserve the parameter/fixed dictionary structure.
    """
    reg_mask = jnp.asarray(regularization_mask, dtype=bool)
    proj_mask = jnp.asarray(projection_mask, dtype=bool)
    if reg_mask.ndim != 1 or reg_mask.size == 0 or reg_mask.shape != proj_mask.shape:
        raise ValueError("Constraint masks must be nonempty 1D arrays of equal shape.")

    def scheduled_regularizer(params, fixed):
        return jax.lax.cond(
            reg_mask[fixed[STEP_KEY]],
            regularizer,
            # zeros_like preserves scalar dtype; inactive penalty work is dead
            # code eliminated by XLA, including its parameter derivatives.
            lambda p, f: jnp.zeros_like(regularizer(p, f)),
            params,
            fixed,
        )

    def scheduled_projection(params, fixed):
        step = fixed[STEP_KEY]
        params, fixed = jax.lax.cond(
            proj_mask[step],
            projection,
            lambda p, f: (p, f),
            params,
            fixed,
        )
        return params, {**fixed, STEP_KEY: step + 1}

    return scheduled_regularizer, jax.jit(scheduled_projection)
