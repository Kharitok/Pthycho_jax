import jax
import jax.numpy as jnp
import numpy as np
from ptycho_jax.DM.init import initialize_state,make_data
from ptycho_jax.DM.stages import forward_from_stages
from ptycho_jax.DM.metrics import amplitude_metrics
from ptycho_jax.DM.handoff import to_ad_parameters,from_ad_parameters,reference_point_model,evaluate_ad,refine_ad
from ptycho_jax.DM.types import DMState
from ptycho_jax.DM.stages import exit_waves


def test_C11_1_to_C11_4_exact_handoff(small_case):
    g,p,o,a=small_case
    state=DMState(exit_waves(p,o,a,g),p,o,a,jnp.array(0))
    data=make_data(forward_from_stages(p,o,a,g)*1.1,jnp.ones((8,10),bool))
    params,fixed=to_ad_parameters(state,g)
    assert 'scan_mistakes' not in params
    assert 'scan_mistakes' in fixed
    result=evaluate_ad(params,fixed,g,data,batch_size=3)
    expected=amplitude_metrics(state.psi,data)['model_mse']
    np.testing.assert_allclose(result['loss'],expected,rtol=1e-12)
    pp,oo,aa=from_ad_parameters(params)
    np.testing.assert_allclose(forward_from_stages(pp,oo,aa,g),forward_from_stages(p,o,a,g),rtol=1e-12,atol=1e-13)


def test_C11_5_AD_descends_and_positions_frozen(small_case):
    g,p,o,a=small_case
    data=make_data(forward_from_stages(p,o,a,g),jnp.ones((8,10),bool))
    start=initialize_state(np.asarray(p[0]),2,data,g,seed=2)
    params,fixed=to_ad_parameters(start,g)
    original={k:np.array(v) for k,v in fixed.items()}
    result,history,metrics=refine_ad(params,fixed,g,data,n_steps=8,batch_size=3,
                                    learning_rates={'sample':.003,'probe_modes':.003,'modal_weights':.0003})
    assert metrics['final_loss']<metrics['initial_loss']
    for key,value in original.items():np.testing.assert_array_equal(fixed[key],value)
    assert set(result)=={'sample','probe_modes','modal_weights'}
    assert len(history)==8
