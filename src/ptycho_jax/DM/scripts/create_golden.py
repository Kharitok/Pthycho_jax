"""Create seeded complex128 reference data; invoke on CPU with PYTHONPATH=src."""
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from ptycho_jax.DM.tests.synthetic import make_case

if __name__=='__main__':
    case=make_case('D',frame=32,side=3,seed=23,dtype=np.complex128)
    g=case['geom']
    path=Path(__file__).parents[1]/'tests/golden/ref_small.npz'
    np.savez_compressed(path,probe=np.asarray(case['probe']),obj=np.asarray(case['obj']),
        a=np.asarray(case['a']),n=np.asarray(g.n),delta=np.asarray(g.delta),
        intensity=np.asarray(case['truth_intensity']),canvas_shape=g.canvas_shape,
        frame_shape=g.frame_shape,margin=g.margin,seed=23)
    print(path)
