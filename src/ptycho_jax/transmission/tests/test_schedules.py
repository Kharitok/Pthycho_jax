"""Constraint schedules must preserve optimizer state and exact step timing."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ptycho_jax.transmission.optim_loop import run_optimization_loop


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        (None, [0, 1, 2, 3, 4, 5]),
        (("none",), []),
        (("all",), [0, 1, 2, 3, 4, 5]),
        (("1:5:2",), [1, 3]),
        (("0:2", "4:"), [0, 1, 4, 5]),
        ((":3", "2", "5"), [0, 1, 2, 5]),
    ],
)
def test_schedule_indices(specs, expected):
    schedules = importlib.import_module("ptycho_jax.transmission.schedules")
    np.testing.assert_array_equal(
        np.flatnonzero(schedules.schedule_mask(specs, 6)), expected
    )


@pytest.mark.parametrize(
    "specs",
    [
        ("-1",),
        ("0:7",),
        ("4:2",),
        ("1:4:0",),
        ("none", "1"),
        ("6",),
        ("",),
        ("1:2:3:4",),
    ],
)
def test_invalid_schedule_rejected(specs):
    schedules = importlib.import_module("ptycho_jax.transmission.schedules")
    with pytest.raises(ValueError):
        schedules.schedule_mask(specs, 6)


def test_scheduled_constraints_keep_one_adam_state_and_count_updates_not_batches():
    schedules = importlib.import_module("ptycho_jax.transmission.schedules")
    params = {"x": jnp.array(0.3, dtype=jnp.float32)}
    fixed = {schedules.STEP_KEY: jnp.array(0, dtype=jnp.int32)}
    regularizer = lambda p, f: 3 * p["x"] ** 2
    projection = lambda p, f: ({"x": jnp.minimum(p["x"], 0.2)}, f)
    reg, project = schedules.schedule_constraints(
        regularizer,
        projection,
        np.array([False, True, False, True]),
        np.array([False, False, True, False]),
    )
    loss = jax.jit(
        jax.value_and_grad(
            lambda p, f, i, measured, mask: (p["x"] - 2) ** 2 + reg(p, f)
        )
    )
    optimizer = optax.adam(0.1)
    final, fixed_out, state_out, history = run_optimization_loop(
        params,
        optimizer.init(params),
        optimizer,
        loss,
        fixed,
        np.ones((5, 2, 2), dtype=np.float32),
        jnp.ones((2, 2), dtype=bool),
        "accumulate_full_pass",
        4,
        2,
        projection_fn=project,
    )
    # Independent explicit updates: TV on steps 1/3, clamp after update 2.
    expected, state = params, optimizer.init(params)
    objectives = []
    for step in range(4):
        x = expected["x"]
        objectives.append(float((x - 2) ** 2 + (3 * x * x if step in (1, 3) else 0)))
        derivative = 2 * (x - 2) + (6 * x if step in (1, 3) else 0)
        updates, state = optimizer.update({"x": derivative}, state, expected)
        expected = optax.apply_updates(expected, updates)
        if step == 2:
            expected = {"x": jnp.minimum(expected["x"], 0.2)}
    np.testing.assert_allclose(final["x"], expected["x"], atol=1e-6)
    np.testing.assert_allclose(history, objectives, atol=1e-6)
    for actual, wanted in zip(
        jax.tree.leaves(state_out), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_allclose(actual, wanted, atol=1e-6)
    assert int(fixed_out[schedules.STEP_KEY]) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regularization_schedule": ("400:401",)},
        {"projection_schedule": ("-2:",)},
        {"z_samples": 8},
        {"padding_factor": 0},
        {"half_range_m": float("nan")},
    ],
)
def test_pipeline_validates_schedule_and_diagnostics_before_output(kwargs):
    import pipeline_nanomax_upd_v2 as pipeline

    with pytest.raises(ValueError):
        pipeline.ReconstructionConfig(**kwargs)
