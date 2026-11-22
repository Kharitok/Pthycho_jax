from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from ptycho_jax.DM.config import DMConfig
from ptycho_jax.DM.conventions import fft2c
from ptycho_jax.DM.geometry import make_geometry
from ptycho_jax.DM.stages import exit_waves
from ptycho_jax.DM.projections import fourier_projection,solve_object,solve_probe,solve_weights,overlap_projection
from ptycho_jax.DM.operators import coverage_map


def test_C4_1_to_C4_4_C4_8_projection(small_case):
    g,p,o,a=small_case
    psi=exit_waves(p,o,a,g)
    root=jnp.sqrt(jnp.sum(abs(fft2c(psi*.7+.1j))**2,axis=1))
    mask=jnp.ones(root.shape,dtype=bool).at[:,0,:].set(False)
    result,stats=fourier_projection(psi,root,mask,1e-14)
    twice,_=fourier_projection(result,root,mask,1e-14)
    np.testing.assert_allclose(twice,result,rtol=1e-12,atol=3e-15)
    f,original=fft2c(result),fft2c(psi)
    np.testing.assert_allclose(jnp.sum(abs(f)**2,axis=1)[mask],(root**2)[mask],rtol=1e-12)
    np.testing.assert_allclose(f[:, :,0,:],original[:,:,0,:],atol=3e-15)
    ratio=f[:,0]/f[:,1]
    np.testing.assert_allclose(ratio[mask],(original[:,0]/original[:,1])[mask],rtol=1e-12)
    expected=jnp.sum(jnp.where(mask,root**2,jnp.sum(abs(original)**2,axis=1)))
    np.testing.assert_allclose(jnp.sum(abs(result)**2),expected,rtol=1e-12)


def test_C4_5_zeros_C4_6_fixed_C4_7_unitary(small_case):
    g,p,o,a=small_case
    psi=exit_waves(p,o,a,g)
    root=jnp.sqrt(jnp.sum(abs(fft2c(psi))**2,axis=1))
    mask=jnp.ones(root.shape,bool)
    np.testing.assert_allclose(fourier_projection(psi,root,mask,1e-14)[0],psi,atol=3e-15)
    zero=fourier_projection(jnp.zeros_like(psi),root,mask,1e-14)[0]
    assert np.isfinite(zero).all()
    np.testing.assert_allclose(jnp.sum(abs(fft2c(zero))**2,axis=1),root**2,rtol=1e-12,atol=1e-14)
    u=jnp.array([[1,1j],[1j,1]])/jnp.sqrt(2.)
    mix=lambda v:jnp.einsum('kl,jlyx->jkyx',u,v)
    np.testing.assert_allclose(fourier_projection(mix(psi*.8),root,mask,1e-14)[0],mix(fourier_projection(psi*.8,root,mask,1e-14)[0]),atol=3e-15)


def test_C5_2_C5_3_C5_4_exact_blocks_and_stationarity(small_case):
    g,p,o,a=small_case
    cfg=DMConfig(object_cg_iterations=250,object_cg_tolerance=1e-13,alpha_object=1e-3,alpha_probe=1e-3)
    target=exit_waves(p,o,a,g)
    old=o*(1+.05j)+.1
    new,alpha=solve_object(target,p,old,a,g,cfg)
    objective=lambda v:.5*jnp.sum(abs(target-exit_waves(p,v,a,g))**2)+.5*alpha*jnp.sum(abs(v-old)**2)
    grad=jax.grad(objective)
    assert jnp.linalg.norm(grad(new))/jnp.linalg.norm(grad(old))<1e-8
    assert objective(new)<=objective(old)
    probe_old=p+.1+.05j
    probe_new,alpha=solve_probe(target,probe_old,o,a,g,cfg)
    loss=lambda v:.5*jnp.sum(abs(target-exit_waves(v,o,a,g))**2)+.5*jnp.sum(alpha[:,None,None]*abs(v-probe_old)**2)
    grad=jax.grad(loss)
    assert jnp.linalg.norm(grad(probe_new))/jnp.linalg.norm(grad(probe_old))<1e-8


def test_C5_5_C5_7_real_nonnegative_weights_KKT(small_case):
    g,p,o,a=small_case
    u=exit_waves(p,o,jnp.ones_like(a),g)
    target=u*jnp.array([[1.,-.4],[.7,2.],[.2,-.1],[3.,.3]])[:,:,None,None]
    got=solve_weights(target,p,o,a,g,2,eps_relative=0.)
    gradient=jax.grad(lambda v:.5*jnp.sum(abs(target-v[:,:,None,None]*u)**2))(got)
    assert np.all(np.asarray(got)>=0)
    assert jnp.max(jnp.abs(jnp.where(got>0,gradient,0)))<1e-10
    assert jnp.min(jnp.where(got==0,gradient,0))>=-1e-10
    global_a=solve_weights(target,p,o,a,g,1,eps_relative=0.)
    np.testing.assert_array_equal(global_a,jnp.broadcast_to(global_a[:1],global_a.shape))


def test_C5_1_fixed_point_C5_8_frozen_gates_C5_13_no_retrace(small_case):
    g,p,o,a=small_case
    target=exit_waves(p,o,a,g)
    cfg=DMConfig(n_inner=2,object_cg_iterations=100,object_cg_tolerance=1e-13)
    covered=coverage_map(g)>0
    calls=[]
    def run(probe_gate,weight_gate):
        calls.append(1)
        return overlap_projection(target,p,o,a,g,cfg,probe_gate,weight_gate,covered)
    fn=jax.jit(run)
    out=fn(jnp.array(False),jnp.array(0))
    np.testing.assert_allclose(out[0],target,rtol=1e-10,atol=1e-12)
    np.testing.assert_array_equal(out[1],p)
    np.testing.assert_array_equal(out[3],a)
    fn(jnp.array(True),jnp.array(2))
    assert len(calls)==1

def test_C6_1_C6_2_C6_3_gauge_preserves_fields(small_case):
    from ptycho_jax.DM.gauge import normalize_all
    g,p,o,a=small_case
    pp,oo,aa=normalize_all(p,o,a,coverage_map(g)>0)
    np.testing.assert_allclose(exit_waves(pp,oo,aa,g),exit_waves(p,o,a,g),rtol=1e-12,atol=2e-15)
    np.testing.assert_allclose(jnp.sum(abs(pp)**2,axis=(-2,-1)),1,rtol=1e-12)
    covered=coverage_map(g)>0
    np.testing.assert_allclose(jnp.mean(abs(oo)[covered]),1,rtol=1e-12)
    assert abs(jnp.angle(jnp.mean(oo[covered])))<1e-12
