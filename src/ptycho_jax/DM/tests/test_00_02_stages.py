import dataclasses
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ptycho_jax.DM.conventions import fft2c,ifft2c,weights_to_a,a_to_weights_param,split_positions
from ptycho_jax.DM.geometry import make_geometry
from ptycho_jax.DM.stages import exit_waves,forward_from_stages
from reference import reference_intensity

def test_C0_1_parseval(small_case):
    g,p,o,a=small_case
    psi=exit_waves(p,o,a,g)
    np.testing.assert_allclose(forward_from_stages(p,o,a,g).sum((-2,-1)),(abs(psi)**2).sum((1,2,3)),rtol=1e-12)
    np.testing.assert_allclose(ifft2c(fft2c(psi)),psi,atol=2e-15)

def test_C0_3_analytic_gaussian(small_case):
    g,_,o,a=small_case
    g=make_geometry(g.n,np.zeros_like(g.delta),g.canvas_shape,g.frame_shape,g.margin)
    y,x=jnp.indices(g.frame_shape)
    p=jnp.exp(-((y-4)**2+(x-5)**2)/8)[None].astype(jnp.complex128)
    result=forward_from_stages(p,jnp.ones_like(o),jnp.ones((4,1)),g)
    np.testing.assert_allclose(result,jnp.broadcast_to(abs(fft2c(p[0]))**2,result.shape),atol=2e-15)

def test_C0_5_weights_and_C2_5_position_roundtrips():
    a=jnp.array([[.2,2.],[3.,.5]])
    np.testing.assert_allclose(weights_to_a(a_to_weights_param(a)),a,rtol=1e-14)
    n,d=split_positions(np.array([[-.2,1.7],[3.4,-2.1]]))
    np.testing.assert_allclose(n+d,[[-.2,1.7],[3.4,-2.1]],atol=1e-15)
    assert np.all((d>=0)&(d<1))
    with pytest.raises(ValueError): weights_to_a(jnp.ones((2,2,2)))

@pytest.mark.parametrize('integer',[False,True])
def test_C1_1_C2_1_C2_2_exact_reference(small_case,integer):
    g,p,o,a=small_case
    if integer:g=make_geometry(g.n,np.zeros_like(g.delta),g.canvas_shape,g.frame_shape,g.margin)
    np.testing.assert_allclose(forward_from_stages(p,o,a,g),reference_intensity(p,o,a,g),rtol=1e-12,atol=1e-13)

def test_C1_2_gradients_match_reference(small_case):
    g,p,o,a=small_case
    target=jnp.arange(4*8*10,dtype=jnp.float64).reshape(4,8,10)/200
    def loss(fn,p,o,a):return jnp.mean((jnp.sqrt(fn(p,o,a,g)+1e-12)-target)**2)
    ours=jax.grad(lambda p,o,a:loss(forward_from_stages,p,o,a),argnums=(0,1,2))(p,o,a)
    ref=jax.grad(lambda p,o,a:loss(reference_intensity,p,o,a),argnums=(0,1,2))(p,o,a)
    for got,want in zip(ours,ref):np.testing.assert_allclose(got,want,rtol=1e-12,atol=1e-14)

def test_C1_3_jit_vmap_C1_4_dtype(small_case):
    g,p,o,a=small_case
    out=jax.jit(forward_from_stages)(p.astype(jnp.complex64),o.astype(jnp.complex64),a.astype(jnp.float32),g)
    assert out.dtype==jnp.float32
    two=jax.vmap(lambda p:forward_from_stages(p,o,a,g))(jnp.stack([p,p]))
    np.testing.assert_array_equal(two[0],two[1])

def test_C2_3_invalid_crop_rejected(small_case):
    g,*_=small_case
    with pytest.raises(ValueError,match='crop'):make_geometry([[19,0]],[[0.,0.]],g.canvas_shape,g.frame_shape,g.margin)

def test_C0_4_integer_crop_sign(small_case):
    from ptycho_jax.DM.stages import object_patches
    g,*_=small_case
    g=make_geometry([[0,0],[1,0]],[[0.,0.],[0.,0.]],g.canvas_shape,g.frame_shape,g.margin)
    obj=jnp.zeros(g.canvas_shape,dtype=jnp.complex128).at[5,5].set(1)
    patches=object_patches(obj,g)
    assert np.unravel_index(np.argmax(abs(patches[0])),g.frame_shape)==(3,3)
    assert np.unravel_index(np.argmax(abs(patches[1])),g.frame_shape)==(2,3)


def test_C0_2_golden_reference_and_reproducibility():
    path=Path(__file__).parent/'golden/ref_small.npz'
    with np.load(path) as f:
        g=make_geometry(f['n'],f['delta'],f['canvas_shape'],f['frame_shape'],int(f['margin']))
        p,o,a=map(jnp.asarray,(f['probe'],f['obj'],f['a']))
        first=reference_intensity(p,o,a,g)
        second=reference_intensity(p,o,a,g)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_allclose(first,f['intensity'],rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(forward_from_stages(p,o,a,g),f['intensity'],rtol=1e-12,atol=1e-10)
