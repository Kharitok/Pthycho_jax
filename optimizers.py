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
