"""
functions for optimization loop
"""

from typing import Any, Callable, Dict, Literal, Tuple

import jax
import jax.numpy as jnp
import optax

PyTree = Any

TrainMode = Literal[
    "accumulate_full_pass",
    "sequential_no_repeats",
    "random_with_replacement",
]


def prepare_opt_step(
    optimizer: optax.GradientTransformation, loss_and_grad_fn: Callable
) -> Callable:
    """
    Prepares an optimization step function that computes the loss and gradients,
    applies the optimizer update, and returns the new parameters and optimizer state.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        The Optax optimizer to use for parameter updates.
    loss_and_grad_fn : Callable
        A function that computes the loss and gradients given parameters.

    Returns
    -------
    Callable
        A function that performs a single optimization step.
    """

    @jax.jit
    def opt_step(
        params: PyTree,
        opt_state: optax.OptState,
        non_diff_params: Dict[str, Any],
        batch_idx: jnp.ndarray,
        batch_measured: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[PyTree, optax.OptState, jnp.ndarray]:
        loss, grads = loss_and_grad_fn(
            params,
            non_diff_params,
            batch_idx,
            batch_measured,
            mask,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    return opt_step


def run_optimization_loop(
    params: PyTree,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    loss_and_grad_fn: Callable,
    non_diff_params: Dict[str, Any],
    measured_batch_pool: jnp.ndarray,
    mask: jnp.ndarray,
    mode: TrainMode,
    n_steps: int,
    batch_size: int,
    seed: int = 0,
    shuffle_each_epoch: bool = True,
) -> Tuple[PyTree, optax.OptState, jnp.ndarray]:
    """
    Runs optimization with configurable gradient/update modes.

    Modes
    -----
    accumulate_full_pass:
        Uses all positions exactly once per epoch, accumulates batch gradients, and
        performs a single optimizer update per epoch.
    sequential_no_repeats:
        Uses all positions exactly once per epoch and updates after every batch.
    random_with_replacement:
        Samples a random batch each step and updates after every batch.

    Parameters
    ----------
    params:
        Optimizable parameter pytree.
    opt_state:
        Optax optimizer state.
    optimizer:
        Optax optimizer transformation.
    loss_and_grad_fn:
        Callable with signature
        (params, non_diff_params, batch_idx, batch_measured, mask) -> (loss, grads).
    non_diff_params:
        Non-optimizable parameter dict.
    measured_batch_pool:
        Full measured data pool indexed by scan position.
    mask:
        Detector/sample mask passed into the loss.
    mode:
        One of the three train modes described above.
    n_steps:
        For full-pass modes this is number of epochs; for random mode it is number
        of update steps.
    batch_size:
        Batch size used for gradient evaluation.
    seed:
        PRNG seed used for shuffling or random sampling.
    shuffle_each_epoch:
        If True, full-pass modes shuffle positions each epoch.

    Returns
    -------
    params:
        Updated parameter pytree.
    opt_state:
        Updated optimizer state.
    loss_history:
        1D array of loss values. In `accumulate_full_pass`, one value per epoch.
        In per-batch modes, one value per update step.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    n_positions = int(measured_batch_pool.shape[0])
    if n_positions < 1:
        raise ValueError("measured_batch_pool must contain at least one position")

    key = jax.random.PRNGKey(seed)
    opt_step = prepare_opt_step(optimizer, loss_and_grad_fn)

    loss_values = []

    if mode == "accumulate_full_pass":
        for _epoch in range(n_steps):
            if shuffle_each_epoch:
                key, subkey = jax.random.split(key)
                order = jax.random.permutation(subkey, n_positions)
            else:
                order = jnp.arange(n_positions, dtype=jnp.int32)

            accum_grads = jax.tree.map(jnp.zeros_like, params)
            weighted_loss_sum = 0.0
            total_items = 0

            for start in range(0, n_positions, batch_size):
                batch_idx = order[start : start + batch_size].astype(jnp.int32)
                batch_measured = measured_batch_pool[batch_idx]

                loss_val, grads = loss_and_grad_fn(
                    params,
                    non_diff_params,
                    batch_idx,
                    batch_measured,
                    mask,
                )

                batch_count = int(batch_idx.shape[0])
                accum_grads = jax.tree.map(
                    lambda g_acc, g: g_acc + g * batch_count,
                    accum_grads,
                    grads,
                )
                weighted_loss_sum += float(loss_val) * batch_count
                total_items += batch_count

            mean_grads = jax.tree.map(lambda g: g / total_items, accum_grads)
            updates, opt_state = optimizer.update(mean_grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            loss_values.append(weighted_loss_sum / total_items)

    elif mode == "sequential_no_repeats":
        for _epoch in range(n_steps):
            if shuffle_each_epoch:
                key, subkey = jax.random.split(key)
                order = jax.random.permutation(subkey, n_positions)
            else:
                order = jnp.arange(n_positions, dtype=jnp.int32)

            for start in range(0, n_positions, batch_size):
                batch_idx = order[start : start + batch_size].astype(jnp.int32)
                batch_measured = measured_batch_pool[batch_idx]

                params, opt_state, loss_val = opt_step(
                    params,
                    opt_state,
                    non_diff_params,
                    batch_idx,
                    batch_measured,
                    mask,
                )
                loss_values.append(float(loss_val))

    elif mode == "random_with_replacement":
        for _step in range(n_steps):
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.randint(
                subkey,
                shape=(batch_size,),
                minval=0,
                maxval=n_positions,
                dtype=jnp.int32,
            )
            batch_measured = measured_batch_pool[batch_idx]

            params, opt_state, loss_val = opt_step(
                params,
                opt_state,
                non_diff_params,
                batch_idx,
                batch_measured,
                mask,
            )
            loss_values.append(float(loss_val))

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected one of: "
            "'accumulate_full_pass', 'sequential_no_repeats', 'random_with_replacement'."
        )

    return params, opt_state, jnp.asarray(loss_values, dtype=jnp.float32)
