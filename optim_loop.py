"""
functions for optimization loop
"""

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

PyTree = Any

TrainMode = Literal[
    "accumulate_full_pass",
    "sequential_no_repeats",
    "random_with_replacement",
]


def prepare_opt_step(
    optimizer: optax.GradientTransformation,
    loss_and_grad_fn: Callable,
    projection_fn: Callable,
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
        non_diff_params: dict[str, Any],
        batch_idx: jnp.ndarray,
        batch_measured: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[PyTree, optax.OptState, dict[str, Any], jnp.ndarray]:
        loss, grads = loss_and_grad_fn(
            params,
            non_diff_params,
            batch_idx,
            batch_measured,
            mask,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params, non_diff_params = projection_fn(params, non_diff_params)

        return params, opt_state, non_diff_params, loss

    return opt_step


def prepare_sharded_opt_step(
    optimizer: optax.GradientTransformation,
    loss_and_grad_fn: Callable,
    projection_fn: Callable,
) -> Callable:
    """Prepares a sharded optimization step that gathers the batch on device."""

    @jax.jit
    def opt_step(
        params: PyTree,
        opt_state: optax.OptState,
        non_diff_params: dict[str, Any],
        measured_batch_pool: jnp.ndarray,
        batch_idx: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[PyTree, optax.OptState, dict[str, Any], jnp.ndarray]:
        batch_measured = measured_batch_pool[batch_idx]
        loss, grads = loss_and_grad_fn(
            params,
            non_diff_params,
            batch_idx,
            batch_measured,
            mask,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params, non_diff_params = projection_fn(params, non_diff_params)

        return params, opt_state, non_diff_params, loss

    return opt_step


def prepare_sharded_accumulate_epoch(
    optimizer: optax.GradientTransformation,
    loss_and_grad_fn: Callable,
    projection_fn: Callable,
) -> Callable:
    """Builds one sharded full-pass epoch with a single optimizer update."""

    @jax.jit
    def accumulate_epoch(
        params: PyTree,
        opt_state: optax.OptState,
        non_diff_params: dict[str, Any],
        measured_batch_pool: jnp.ndarray,
        epoch_batch_idx: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[PyTree, optax.OptState, dict[str, Any], jnp.ndarray]:
        batch_weight = epoch_batch_idx.shape[1]
        init_grads = jax.tree.map(jnp.zeros_like, params)
        init_weighted_loss = jnp.array(0.0, dtype=jnp.float32)

        def scan_step(carry, batch_idx):
            grad_sum, weighted_loss_sum = carry
            batch_measured = measured_batch_pool[batch_idx]
            loss_val, grads = loss_and_grad_fn(
                params,
                non_diff_params,
                batch_idx,
                batch_measured,
                mask,
            )
            grad_sum = jax.tree.map(
                lambda g_acc, g, weight=batch_weight: g_acc + g * weight,
                grad_sum,
                grads,
            )
            weighted_loss_sum = weighted_loss_sum + loss_val * batch_weight
            return (grad_sum, weighted_loss_sum), None

        (accum_grads, weighted_loss_sum), _ = jax.lax.scan(
            scan_step,
            (init_grads, init_weighted_loss),
            epoch_batch_idx,
        )

        total_items = epoch_batch_idx.shape[0] * batch_weight
        mean_grads = jax.tree.map(
            lambda g, denom=total_items: g / denom,
            accum_grads,
        )
        updates, opt_state = optimizer.update(mean_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params, non_diff_params = projection_fn(params, non_diff_params)
        mean_loss = weighted_loss_sum / total_items

        return params, opt_state, non_diff_params, mean_loss

    return accumulate_epoch


def _create_shardings(
    exemplar_pool: jnp.ndarray,
) -> tuple[Mesh, NamedSharding, NamedSharding]:
    devices = np.array(jax.local_devices())
    mesh = Mesh(devices, axis_names=("data",))
    replicated = NamedSharding(mesh, P())
    pool_sharding = NamedSharding(
        mesh,
        P("data", *([None] * (exemplar_pool.ndim - 1))),
    )
    return mesh, replicated, pool_sharding


def _place_sharded_training_state(
    params: PyTree,
    opt_state: optax.OptState,
    non_diff_params: dict[str, Any],
    measured_batch_pool: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[
    PyTree, optax.OptState, dict[str, Any], jnp.ndarray, jnp.ndarray, NamedSharding
]:
    _, replicated, pool_sharding = _create_shardings(measured_batch_pool)
    params = jax.device_put(params, replicated)
    opt_state = jax.device_put(opt_state, replicated)
    non_diff_params = jax.device_put(non_diff_params, replicated)
    measured_batch_pool = jax.device_put(measured_batch_pool, pool_sharding)
    mask = jax.device_put(mask, replicated)
    return params, opt_state, non_diff_params, measured_batch_pool, mask, replicated


def _build_full_pass_schedule(
    n_positions: int,
    n_steps: int,
    batch_size: int,
    key: jax.Array,
    shuffle_each_epoch: bool,
) -> tuple[jnp.ndarray, jax.Array]:
    batches_per_epoch = n_positions // batch_size
    epoch_batches = []

    for _ in range(n_steps):
        if shuffle_each_epoch:
            key, subkey = jax.random.split(key)
            order = jax.random.permutation(subkey, n_positions)
        else:
            order = jnp.arange(n_positions, dtype=jnp.int32)
        epoch_batches.append(order.reshape(batches_per_epoch, batch_size))

    return jnp.stack(epoch_batches).astype(jnp.int32), key


def _build_random_schedule(
    n_positions: int,
    n_steps: int,
    batch_size: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jax.Array]:
    key, subkey = jax.random.split(key)
    batch_schedule = jax.random.randint(
        subkey,
        shape=(n_steps, batch_size),
        minval=0,
        maxval=n_positions,
        dtype=jnp.int32,
    )
    return batch_schedule, key


def _collect_to_host(tree: PyTree) -> PyTree:
    return jax.device_get(tree)


def run_optimization_loop(
    params: PyTree,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    loss_and_grad_fn: Callable,
    non_diff_params: dict[str, Any],
    measured_batch_pool: jnp.ndarray,
    mask: jnp.ndarray,
    mode: TrainMode,
    n_steps: int,
    batch_size: int,
    seed: int = 0,
    shuffle_each_epoch: bool = True,
    use_multigpu: bool = False,
    projection_fn: Callable | None = None,
) -> tuple[PyTree, optax.OptState, jnp.ndarray]:
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
        Batch size used for gradient evaluation. When `use_multigpu=True`, this is
        the global batch size across all devices.
    seed:
        PRNG seed used for shuffling or random sampling.
    shuffle_each_epoch:
        If True, full-pass modes shuffle positions each epoch.
    use_multigpu:
        If True, shard the measurement pool once, replicate params and optimizer
        state once, and run the optimization against those sharded arrays.
    projection_fn:
        Optional projection callback with signature
        (params, non_diff_params) -> (params, non_diff_params).
        If provided, it is applied after each optimizer update.

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
    loss_values = []

    if projection_fn is None:

        def effective_projection_fn(params: PyTree, non_diff_params: dict[str, Any]):
            return params, non_diff_params

    else:
        effective_projection_fn = projection_fn

    if use_multigpu:
        n_devices = jax.local_device_count()
        if n_devices < 2:
            raise ValueError(
                "use_multigpu=True requires at least 2 visible JAX devices"
            )
        if batch_size % n_devices != 0:
            raise ValueError(
                "In multi-GPU mode, batch_size is global and must be divisible by the number of devices"
            )
        if mode != "random_with_replacement" and n_positions % batch_size != 0:
            raise ValueError(
                "In multi-GPU full-pass modes, measured_batch_pool.shape[0] must be divisible by batch_size so batches can stay sharded with one fixed shape"
            )

        (
            params,
            opt_state,
            non_diff_params,
            measured_batch_pool,
            mask,
            replicated_sharding,
        ) = _place_sharded_training_state(
            params,
            opt_state,
            non_diff_params,
            measured_batch_pool,
            mask,
        )

        sharded_opt_step = prepare_sharded_opt_step(
            optimizer,
            loss_and_grad_fn,
            effective_projection_fn,
        )
        sharded_accumulate_epoch = prepare_sharded_accumulate_epoch(
            optimizer,
            loss_and_grad_fn,
            effective_projection_fn,
        )

        if mode == "accumulate_full_pass":
            epoch_schedule, key = _build_full_pass_schedule(
                n_positions,
                n_steps,
                batch_size,
                key,
                shuffle_each_epoch,
            )
            epoch_schedule = jax.device_put(epoch_schedule, replicated_sharding)

            for epoch_idx in range(n_steps):
                params, opt_state, non_diff_params, loss_val = sharded_accumulate_epoch(
                    params,
                    opt_state,
                    non_diff_params,
                    measured_batch_pool,
                    epoch_schedule[epoch_idx],
                    mask,
                )
                loss_values.append(float(jax.device_get(loss_val)))

        elif mode == "sequential_no_repeats":
            epoch_schedule, key = _build_full_pass_schedule(
                n_positions,
                n_steps,
                batch_size,
                key,
                shuffle_each_epoch,
            )
            batch_schedule = epoch_schedule.reshape(-1, batch_size)
            batch_schedule = jax.device_put(batch_schedule, replicated_sharding)

            for step_idx in range(batch_schedule.shape[0]):
                params, opt_state, non_diff_params, loss_val = sharded_opt_step(
                    params,
                    opt_state,
                    non_diff_params,
                    measured_batch_pool,
                    batch_schedule[step_idx],
                    mask,
                )
                loss_values.append(float(jax.device_get(loss_val)))

        elif mode == "random_with_replacement":
            batch_schedule, key = _build_random_schedule(
                n_positions,
                n_steps,
                batch_size,
                key,
            )
            batch_schedule = jax.device_put(batch_schedule, replicated_sharding)

            for step_idx in range(n_steps):
                params, opt_state, non_diff_params, loss_val = sharded_opt_step(
                    params,
                    opt_state,
                    non_diff_params,
                    measured_batch_pool,
                    batch_schedule[step_idx],
                    mask,
                )
                loss_values.append(float(jax.device_get(loss_val)))

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Expected one of: "
                "'accumulate_full_pass', 'sequential_no_repeats', 'random_with_replacement'."
            )

        return (
            _collect_to_host(params),
            _collect_to_host(opt_state),
            jnp.asarray(
                loss_values,
                dtype=jnp.float32,
            ),
        )

    opt_step = prepare_opt_step(
        optimizer,
        loss_and_grad_fn,
        effective_projection_fn,
    )
    if mode == "accumulate_full_pass":
        n_full_batches = n_positions // batch_size
        remainder = n_positions % batch_size

        @jax.jit
        def _accumulate_full_batches(
            params: PyTree,
            non_diff_params: dict[str, Any],
            full_batch_idx: jnp.ndarray,
            measured_batch_pool: jnp.ndarray,
            mask: jnp.ndarray,
        ) -> tuple[PyTree, jnp.ndarray]:
            init_grads = jax.tree.map(jnp.zeros_like, params)
            init_weighted_loss = jnp.array(0.0, dtype=jnp.float32)

            def _scan_step(carry, batch_idx):
                grad_sum, weighted_loss_sum = carry
                batch_measured = measured_batch_pool[batch_idx]
                loss_val, grads = loss_and_grad_fn(
                    params,
                    non_diff_params,
                    batch_idx,
                    batch_measured,
                    mask,
                )
                grad_sum = jax.tree.map(
                    lambda g_acc, g: g_acc + g * batch_size,
                    grad_sum,
                    grads,
                )
                weighted_loss_sum = weighted_loss_sum + loss_val * batch_size
                return (grad_sum, weighted_loss_sum), None

            (accum_grads, weighted_loss_sum), _ = jax.lax.scan(
                _scan_step,
                (init_grads, init_weighted_loss),
                full_batch_idx,
            )
            return accum_grads, weighted_loss_sum

        for _epoch in range(n_steps):
            if shuffle_each_epoch:
                key, subkey = jax.random.split(key)
                order = jax.random.permutation(subkey, n_positions)
            else:
                order = jnp.arange(n_positions, dtype=jnp.int32)

            accum_grads = jax.tree.map(jnp.zeros_like, params)
            weighted_loss_sum = jnp.array(0.0, dtype=jnp.float32)
            total_items = 0

            if n_full_batches > 0:
                full_batch_idx = order[: n_full_batches * batch_size].reshape(
                    n_full_batches, batch_size
                )
                accum_grads, weighted_loss_sum = _accumulate_full_batches(
                    params,
                    non_diff_params,
                    full_batch_idx,
                    measured_batch_pool,
                    mask,
                )
                total_items += n_full_batches * batch_size

            if remainder > 0:
                tail_idx = order[n_full_batches * batch_size :].astype(jnp.int32)
                tail_measured = measured_batch_pool[tail_idx]
                tail_loss, tail_grads = loss_and_grad_fn(
                    params,
                    non_diff_params,
                    tail_idx,
                    tail_measured,
                    mask,
                )
                accum_grads = jax.tree.map(
                    lambda g_acc, g: g_acc + g * remainder,
                    accum_grads,
                    tail_grads,
                )
                weighted_loss_sum = weighted_loss_sum + tail_loss * remainder
                total_items += remainder

            mean_grads = jax.tree.map(
                lambda g, denom=total_items: g / denom,
                accum_grads,
            )
            updates, opt_state = optimizer.update(mean_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params, non_diff_params = effective_projection_fn(params, non_diff_params)

            loss_values.append(float(weighted_loss_sum / total_items))

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

                params, opt_state, non_diff_params, loss_val = opt_step(
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

            params, opt_state, non_diff_params, loss_val = opt_step(
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
