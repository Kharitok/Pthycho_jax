import jax.numpy as jnp
import numpy as np
from ptycho_jax.DM.tests.synthetic import make_case
from ptycho_jax.DM.stages import forward_from_stages
from ptycho_jax.DM.metrics import aligned_object_error,coherence_overlap
from ptycho_jax.DM.operators import coverage_map


def test_C10_1_C10_5_generator_reference_and_seed():
    c=make_case('D',frame=16,side=3,dtype=np.complex128)
    np.testing.assert_allclose(forward_from_stages(c['probe'],c['obj'],c['a'],c['geom']),c['truth_intensity'],rtol=1e-12,atol=1e-10)
    repeat=make_case('D',frame=16,side=3,dtype=np.complex128)
    np.testing.assert_array_equal(c['truth_intensity'],repeat['truth_intensity'])


def test_C10_2_gauge_alignment_C10_3_coherence():
    c=make_case('C',frame=16,side=3,dtype=np.complex128)
    y,x=np.indices(c['obj'].shape)
    changed=c['obj']*(1.3+.2j)*np.exp(1j*(.03*y-.07*x+.3))
    assert aligned_object_error(changed,c['obj'],coverage_map(c['geom'])>0)<1e-12
    p=np.asarray(c['probe'])
    mixed=np.einsum('kl,lyx->kyx',np.linalg.qr(np.array([[1,2j,1],[3j,1,2],[1,3,2j]]))[0],p)
    assert abs(coherence_overlap(mixed,np.ones(3),p,np.ones(3))-1)<1e-12

def test_C10_2_alignment_ignores_uncovered_phase():
    rng=np.random.default_rng(17)
    y,x=np.indices((40,40))
    truth=np.exp(.1j*np.sin(x/4))
    covered=np.zeros((40,40),bool);covered[10:30,10:30]=True
    recovered=2*truth*np.exp(1j*(.15*y-.2*x))
    recovered[~covered]=np.exp(1j*rng.uniform(-np.pi,np.pi,np.sum(~covered)))
    assert aligned_object_error(recovered,truth,covered)<1e-10
