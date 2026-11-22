"""Run with JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from ptycho_jax.DM.tests.synthetic import make_case
from ptycho_jax.DM.init import initialize_state
from ptycho_jax.DM.dm import dm_step
from ptycho_jax.DM.config import DMConfig
from ptycho_jax.DM.sharding import shard_positions

if __name__=='__main__':
    if len(jax.devices())!=4:raise RuntimeError('Start with four virtual CPU devices.')
    c=make_case('D',frame=16,side=4,dtype=np.complex128)
    state=initialize_state(c['probe_guess'],3,c['data'],c['geom'])
    cfg=DMConfig(n_inner=1,probe_start=0,weight_strategy='per_position',object_cg_iterations=12)
    fn=jax.jit(lambda s,d,g:dm_step(s,d,g,cfg))
    reference,_=fn(state,c['data'],c['geom'])
    sharded=shard_positions(state,c['data'],c['geom'])
    got,_=fn(*sharded)
    errors={name:float(np.linalg.norm(np.asarray(x-y))/max(np.linalg.norm(np.asarray(y)),1e-30))
            for name,x,y in zip(reference._fields,got,reference)}
    if max(errors.values())>1e-12:raise AssertionError(errors)
    out=Path('results/dm_benchmark');out.mkdir(parents=True,exist_ok=True)
    (out/'sharding.json').write_text(json.dumps({'devices':4,'relative_errors':errors},indent=2)+'\n')
    print(errors)
