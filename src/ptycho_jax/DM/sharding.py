"""Optional position-axis placement using JAX global-array automatic sharding."""

from dataclasses import replace
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from .types import DMState, Data


def shard_positions(state, data, geom, devices=None):
    """Place J-leading arrays across devices, replicate probe/object.

    Requires J divisible by device count; never drops or pads measurements.
    Existing pure global-array sums/scatters insert the necessary collectives.
    """
    devices = list(jax.devices() if devices is None else devices)
    if len(geom.n) % len(devices):
        raise ValueError("Shot count must be divisible by device count.")
    mesh = Mesh(np.asarray(devices), ("shots",))
    replicate = lambda x: jax.device_put(x, NamedSharding(mesh, P()))
    partition = lambda x: jax.device_put(
        x, NamedSharding(mesh, P("shots", *([None] * (x.ndim - 1))))
    )
    state = DMState(
        partition(state.psi),
        replicate(state.probe),
        replicate(state.obj),
        partition(state.a),
        replicate(state.it),
    )
    data = Data(partition(data.sqrt_I), partition(data.mask), partition(data.counts))
    geom = replace(geom, n=partition(geom.n), delta=partition(geom.delta))
    return state, data, geom
