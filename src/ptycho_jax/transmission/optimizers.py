"""
optimizers and routines for ptychography
"""

from typing import Callable, Dict

import jax
import jax.numpy as jnp
import optax


def conj_grads() -> optax.GradientTransformation:
    """Conjugate grads of complex parameters for correct optimization.

    use as :
    optax.chain(
        conj_grads(),
        optax.multi_transform(transforms, label_fn),
    )
    """
    return optax.stateless(lambda grads, params=None: jax.tree.map(jnp.conj, grads))


def prepare_optimizer(opt_params_per_leaf: Dict) -> optax.GradientTransformation:
    """Prepares an optimizer with different learning rates for different parameter leaves."""
    transforms = {}

    for param_name, opt_params in opt_params_per_leaf.items():
        opt_type = opt_params.get("type", "adam")
        learning_rate = opt_params["learning_rate"]

        if opt_type == "adam":
            transforms[param_name] = optax.adam(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

    def label_fn(params):
        unknown = set(params.keys()) - set(transforms.keys())
        if unknown:
            raise ValueError(
                f"Missing optimizer config for parameters: {sorted(unknown)}"
            )
        return {name: name for name in params.keys()}

    return optax.chain(
        conj_grads(),
        optax.multi_transform(transforms, label_fn),
    )





###
# import math
# import optax

# def estimate_total_updates(mode: str, n_steps: int, n_positions: int, batch_size: int) -> int:
#     if mode == "random_with_replacement":
#         return n_steps
#     if mode == "accumulate_full_pass":
#         return n_steps  # one update per epoch
#     if mode == "sequential_no_repeats":
#         return n_steps * math.ceil(n_positions / batch_size)
#     raise ValueError(f"Unknown mode: {mode}")

# # Match these to your run_optimization_loop call
# mode = "sequential_no_repeats"
# n_steps = 50
# batch_size = 10
# n_positions = measured_pool.shape[0]

# total_updates = estimate_total_updates(mode, n_steps, n_positions, batch_size)

# # Milestones
# warmup_steps = max(10, int(0.05 * total_updates))
# probe_start = int(0.15 * total_updates)
# scan_start = int(0.30 * total_updates)
# scan_ramp = max(10, int(0.10 * total_updates))

# # 1) Sample: warmup -> cosine decay
# sample_sched = optax.join_schedules(
#     schedules=[
#         optax.linear_schedule(
#             init_value=1e-3,
#             end_value=1e-1,
#             transition_steps=warmup_steps,
#         ),
#         optax.cosine_decay_schedule(
#             init_value=1e-1,
#             decay_steps=max(1, total_updates - warmup_steps),
#             alpha=0.05,  # final LR is 5% of peak
#         ),
#     ],
#     boundaries=[warmup_steps],
# )

# # 2) Probe: frozen early -> small exponential decay
# probe_sched = optax.join_schedules(
#     schedules=[
#         optax.constant_schedule(0.0),
#         optax.exponential_decay(
#             init_value=2e-3,
#             transition_steps=max(1, int(0.2 * total_updates)),
#             decay_rate=0.95,
#             staircase=False,
#         ),
#     ],
#     boundaries=[probe_start],
# )

# # 3) Scan mistakes: frozen -> ramp -> decay
# scan_sched = optax.join_schedules(
#     schedules=[
#         optax.constant_schedule(0.0),
#         optax.linear_schedule(
#             init_value=0.0,
#             end_value=1e-2,
#             transition_steps=scan_ramp,
#         ),
#         optax.exponential_decay(
#             init_value=1e-2,
#             transition_steps=max(1, int(0.25 * total_updates)),
#             decay_rate=0.9,
#             staircase=False,
#         ),
#     ],
#     boundaries=[scan_start, scan_start + scan_ramp],
# )

# # 4) Modal weights: keep frozen or tiny LR
# modal_sched = optax.constant_schedule(0.0)

# opt_params_per_leaf = {
#     "sample": {"type": "adam", "learning_rate": sample_sched},
#     "probe_modes": {"type": "adam", "learning_rate": probe_sched},
#     "modal_weights": {"type": "adam", "learning_rate": modal_sched},
#     "scan_mistakes": {"type": "adam", "learning_rate": scan_sched},
# }

# optimizer = prepare_optimizer(opt_params_per_leaf)
# opt_state = optimizer.init(params)