"""Collect independent float64 algebra measurements for the phase reports."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
from pathlib import Path
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from ptycho_jax.DM.geometry import make_geometry
from ptycho_jax.DM.stages import exit_waves,forward_from_stages
from ptycho_jax.DM.operators import object_forward,object_adjoint,probe_forward,probe_adjoint,coverage_map
from ptycho_jax.DM.conventions import hermitian_adjoint
from ptycho_jax.DM.projections import fourier_projection,solve_object,solve_probe
from ptycho_jax.DM.config import DMConfig
from ptycho_jax.DM.gauge import normalize_all
from ptycho_jax.DM.init import make_data,initialize_state
from ptycho_jax.DM.metrics import amplitude_metrics
from ptycho_jax.DM.handoff import to_ad_parameters,evaluate_ad
from ptycho_jax.DM.types import DMState


def relative(x,y):return float(jnp.linalg.norm(x-y)/jnp.maximum(jnp.linalg.norm(y),1e-30))

if __name__=='__main__':
    root=Path(__file__).parents[1]
    with np.load(root/'tests/golden/ref_small.npz') as f:
        g=make_geometry(f['n'],f['delta'],f['canvas_shape'],f['frame_shape'],int(f['margin']))
        p,o,a=map(jnp.asarray,(f['probe'],f['obj'],f['a']))
        expected=jnp.asarray(f['intensity'])
    psi=exit_waves(p,o,a,g);target=psi*(.8+.1j)
    result={'reference_intensity_relative_error':relative(forward_from_stages(p,o,a,g),expected)}
    for name,value,forward,adj in (
        ('object',o,lambda x:object_forward(x,p,a,g),lambda y:object_adjoint(y,p,a,g)),
        ('probe',p,lambda x:probe_forward(x,o,a,g),lambda y:probe_adjoint(y,o,a,g))):
        result[name+'_dot_relative_error']=float(abs(jnp.vdot(forward(value),target)-jnp.vdot(value,adj(target)))/(jnp.linalg.norm(forward(value))*jnp.linalg.norm(target)))
        result[name+'_linear_transpose_relative_error']=relative(adj(target),hermitian_adjoint(forward,value)(target))
    data=make_data(expected*1.2,jnp.ones(g.frame_shape,bool))
    pf,_=fourier_projection(psi,data.sqrt_I,data.mask,1e-14)
    pf2,_=fourier_projection(pf,data.sqrt_I,data.mask,1e-14)
    result['PF_idempotence_relative_error']=relative(pf2,pf)
    result['PF_intensity_relative_error']=relative(jnp.sum(abs(jnp.fft.fft2(pf,norm='ortho'))**2,axis=1),data.sqrt_I**2)
    pp,oo,aa=normalize_all(p,o,a,coverage_map(g)>0)
    result['gauge_field_relative_error']=relative(exit_waves(pp,oo,aa,g),psi)
    cfg=DMConfig(object_cg_iterations=500,object_cg_tolerance=1e-13)
    old=o+.05+.02j
    new,alpha=solve_object(psi,p,old,a,g,cfg)
    fn=lambda v:.5*jnp.sum(abs(psi-exit_waves(p,v,a,g))**2)+.5*alpha*jnp.sum(abs(v-old)**2)
    grad=jax.grad(fn);result['object_LS_gradient_ratio']=float(jnp.linalg.norm(grad(new))/jnp.linalg.norm(grad(old)))
    oldp=p+.001+.002j
    newp,alpha=solve_probe(psi,oldp,o,a,g,cfg)
    fn=lambda v:.5*jnp.sum(abs(psi-exit_waves(v,o,a,g))**2)+.5*jnp.sum(alpha[:,None,None]*abs(v-oldp)**2)
    grad=jax.grad(fn);result['probe_LS_gradient_ratio']=float(jnp.linalg.norm(grad(newp))/jnp.linalg.norm(grad(oldp)))
    state=DMState(psi,p,o,a,jnp.array(0,jnp.int32))
    params,fixed=to_ad_parameters(state,g)
    reference=evaluate_ad(params,fixed,g,data)
    our=float(amplitude_metrics(psi,data)['model_mse'])
    result['handoff_MSE_relative_error']=abs(reference['loss']-our)/our
    (root/'reports/measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
