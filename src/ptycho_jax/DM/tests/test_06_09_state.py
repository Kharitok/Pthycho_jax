from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ptycho_jax.DM.config import DMConfig
from ptycho_jax.DM.stages import exit_waves,forward_from_stages
from ptycho_jax.DM.init import initialize_modes,initialize_state,make_data
from ptycho_jax.DM.dm import DMState,dm_step,dm_step_general,run_dm


def test_C7_1_to_C7_4_initialization(small_case):
    g,p,o,a=small_case
    p0=np.asarray(p[0])
    modes=initialize_modes(p0,2,seed=17)
    flat=modes.reshape(2,-1)
    np.testing.assert_allclose(flat@flat.conj().T,np.eye(2),atol=1e-12)
    np.testing.assert_allclose(modes[0],p0/np.linalg.norm(p0),atol=1e-12)
    np.testing.assert_array_equal(modes,initialize_modes(p0,2,seed=17))
    assert not np.allclose(modes[1],initialize_modes(p0,2,seed=18)[1])
    measured=forward_from_stages(p,o,a,g)
    data=make_data(measured,jnp.ones((8,10),bool))
    state=initialize_state(p0,2,data,g,seed=17)
    predicted=forward_from_stages(state.probe,state.obj,state.a,g)
    np.testing.assert_allclose(predicted.sum((-2,-1)),measured.sum((-2,-1)),rtol=1e-12)


def test_C8_1_fixed_point_C8_5_general_beta_one(small_case):
    g,p,o,a=small_case
    data=make_data(forward_from_stages(p,o,a,g),jnp.ones((8,10),bool))
    state=DMState(exit_waves(p,o,a,g),p,o,a,jnp.array(0,jnp.int32))
    cfg=DMConfig(n_inner=2,probe_start=0,global_weights_start=0,per_position_weights_start=0,
                 object_cg_iterations=100,object_cg_tolerance=1e-13,weight_eps_relative=0.)
    got,metrics=dm_step(state,data,g,cfg)
    np.testing.assert_allclose(got.psi,state.psi,rtol=1e-10,atol=1e-12)
    general,_=dm_step_general(state,data,g,cfg)
    np.testing.assert_allclose(general.psi,got.psi,rtol=1e-12,atol=1e-12)
    assert metrics['model_r']<1e-20


@pytest.mark.parametrize('chunk',[0,3])
def test_C8_2_C8_3_C8_9_trace_scan_and_gates(small_case,chunk):
    g,p,o,a=small_case
    data=make_data(forward_from_stages(p,o,a,g),jnp.ones((8,10),bool))
    initial=initialize_state(np.asarray(p[0]),2,data,g,seed=1)
    cfg=DMConfig(n_inner=1,probe_start=1,global_weights_start=2,per_position_weights_start=3,chunk_size=chunk)
    traces=[]
    def step(s,d,g):
        traces.append(1)
        return dm_step(s,d,g,cfg)
    fn=jax.jit(step)
    s=initial
    python_metrics=[]
    for i in range(4):
        old=s
        s,m=fn(s,data,g)
        python_metrics.append(m)
        if i==0:
            np.testing.assert_array_equal(s.probe,old.probe)
            np.testing.assert_array_equal(s.a,old.a)
    assert len(traces)==1
    result,history=run_dm(initial,data,g,cfg,n_iter=4,log_every=4,donate=False)
    np.testing.assert_allclose(result.psi,s.psi,rtol=1e-12,atol=1e-12)
    assert history['model_mse'].shape==(4,)


def test_C9_1_C9_2_C9_3_metrics_match_independent_values(small_case):
    g,p,o,a=small_case
    data=make_data(forward_from_stages(p,o,a,g)*1.2,jnp.ones((8,10),bool))
    s=DMState(exit_waves(p,o,a,g),p,o,a,jnp.array(0,jnp.int32))
    out,m=dm_step(s,data,g,DMConfig(n_inner=1))
    prediction=forward_from_stages(out.probe,out.obj,out.a,g)
    expected=jnp.mean((jnp.sqrt(prediction+1e-20)-data.sqrt_I)**2)
    np.testing.assert_allclose(m['model_mse'],expected,rtol=1e-12)
    from ptycho_jax.DM.projections import fourier_projection
    pf,_=fourier_projection(s.psi,data.sqrt_I,data.mask,1e-14)
    np.testing.assert_allclose(m['update_norm'],jnp.sum(abs(out.psi-s.psi)**2)/jnp.sum(abs(pf)**2),rtol=1e-12)


def test_C8_4_independent_numpy_DM(small_case):
    from ptycho_jax.DM.geometry import make_geometry
    from numpy_reference import iterate
    g,p,o,a=small_case
    g=make_geometry(g.n,np.zeros_like(g.delta),g.canvas_shape,g.frame_shape,g.margin)
    data=make_data(forward_from_stages(p,o,a,g),jnp.ones((8,10),bool))
    initial_obj=jnp.ones_like(o)
    initial=DMState(exit_waves(p,initial_obj,a,g),p,initial_obj,a,jnp.array(0,jnp.int32))
    cfg=DMConfig(n_inner=2,probe_start=0,weight_strategy='fixed')
    expected=iterate(np.asarray(initial.psi),np.asarray(p),np.asarray(initial_obj),np.asarray(a),
                     np.asarray(g.n),g.margin,np.asarray(data.sqrt_I),20)
    state,_=run_dm(initial,data,g,cfg,20,log_every=20,donate=False)
    for got,want in zip((state.psi,state.probe,state.obj,state.a),expected):
        np.testing.assert_allclose(got,want,rtol=1e-10,atol=1e-10)


def test_C6_6_zero_mode_freezes_and_is_logged(small_case):
    g,p,o,a=small_case
    a=a.at[:,1].set(0)
    state=DMState(exit_waves(p,o,a,g),p,o,a,jnp.array(0,jnp.int32))
    data=make_data(forward_from_stages(p,o,a,g),jnp.ones((8,10),bool))
    cfg=DMConfig(n_inner=1,probe_start=0,weight_strategy='fixed')
    result,m=dm_step(state,data,g,cfg)
    np.testing.assert_array_equal(result.probe[1],p[1])
    assert m['inactive_modes']==1


def test_C8_6_general_beta_persistent_first_projection(small_case):
    from ptycho_jax.DM.projections import fourier_projection,overlap_projection
    from ptycho_jax.DM.operators import coverage_map
    g,p,o,a=small_case
    data=make_data(forward_from_stages(p,o,a,g)*1.1,jnp.ones((8,10),bool))
    state=DMState(exit_waves(p,o,a,g),p,o,a,jnp.array(0,jnp.int32))
    cfg=DMConfig(beta=.7,n_inner=1,probe_start=0,weight_strategy='per_position')
    pf,_=fourier_projection(state.psi,data.sqrt_I,data.mask,cfg.fourier_eps_relative*jnp.sqrt(jnp.mean(data.sqrt_I**2)))
    expected=overlap_projection((1+1/cfg.beta)*pf-state.psi/cfg.beta,p,o,a,g,cfg,jnp.array(True),jnp.array(2),coverage_map(g)>0)
    result,_=dm_step(state,data,g,cfg)
    for got,want in zip((result.probe,result.obj,result.a),expected[1:]):
        np.testing.assert_allclose(got,want,rtol=1e-12,atol=1e-12)


def test_C12_1_chunked_full_step(small_case):
    g,p,o,a=small_case
    data=make_data(forward_from_stages(p,o,a,g)*1.1,jnp.ones((8,10),bool))
    state=DMState(exit_waves(p,o,a,g),p,o,a,jnp.array(0,jnp.int32))
    cfg=DMConfig(n_inner=1,probe_start=0,weight_strategy='per_position')
    full,_=dm_step(state,data,g,cfg)
    chunked,_=dm_step(state,data,g,replace(cfg,chunk_size=3))
    for got,want in zip(chunked,full):np.testing.assert_allclose(got,want,rtol=1e-12,atol=1e-12)
