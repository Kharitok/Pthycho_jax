"""Compile/time DM on three sizes, integer/fractional positions and chunked adjoints."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from dataclasses import replace
from pathlib import Path
import argparse,json,time
import jax
import numpy as np
from ptycho_jax.DM.tests.synthetic import make_case
from ptycho_jax.DM.geometry import make_geometry
from ptycho_jax.DM.config import DMConfig
from ptycho_jax.DM.init import initialize_state
from ptycho_jax.DM.dm import dm_step


def benchmark(output):
    rows=[]
    for frame,side in ((16,4),(32,5),(64,7)):
        c=make_case('D',frame=frame,side=side)
        for integer in (False,True):
            g=c['geom']
            if integer:g=make_geometry(g.n,np.zeros_like(g.delta),g.canvas_shape,g.frame_shape,g.margin)
            state=initialize_state(c['probe_guess'],3,c['data'],g)
            for chunk in (0,8):
                cfg=DMConfig(n_inner=1,object_cg_iterations=3,chunk_size=chunk,probe_start=0,weight_strategy='per_position')
                fn=jax.jit(lambda s,d,gg:dm_step(s,d,gg,cfg))
                compiled=fn.lower(state,c['data'],g).compile()
                out=compiled(state,c['data'],g);jax.block_until_ready(out)
                start=time.perf_counter()
                for _ in range(10):out=compiled(state,c['data'],g)
                jax.block_until_ready(out)
                elapsed=(time.perf_counter()-start)/10
                memory=compiled.memory_analysis()
                total=memory.argument_size_in_bytes+memory.output_size_in_bytes+memory.temp_size_in_bytes-memory.alias_size_in_bytes
                rows.append({'frame':frame,'positions':len(g.n),'K':3,'integer_only':integer,'chunk':chunk,
                    'seconds_per_iteration':elapsed,'compiled_buffer_estimate_MiB':total/2**20,
                    'temporary_MiB':memory.temp_size_in_bytes/2**20})
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    (path/'benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    text=['# DM compile/memory benchmark','','One inner sweep, three CG iterations. Same three-mode problem sizes. '
          'The memory estimate is compiled arguments + outputs + temporaries − aliases, not an independently sampled process peak. '
          'Chunking bounds the backprojection intermediates; the complete DM iterate is still resident.','',
          '| Frame | Shots | Integer | Adjoint chunk | Seconds/iteration | Compiled buffers (MiB) |',
          '| --- | --- | --- | --- | --- | --- |']
    for r in rows:text.append(f"| {r['frame']} | {r['positions']} | {r['integer_only']} | {r['chunk']} | {r['seconds_per_iteration']:.6g} | {r['compiled_buffer_estimate_MiB']:.3f} |")
    (path/'benchmark.md').write_text('\n'.join(text)+'\n')
    print(path/'benchmark.md')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results/dm_benchmark')
    benchmark(p.parse_args().output)
