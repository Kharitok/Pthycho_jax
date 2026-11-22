import jax
import jax.numpy as jnp
import numpy as np
from ptycho_jax.DM.conventions import hermitian_adjoint
from ptycho_jax.DM.geometry import make_geometry
from ptycho_jax.DM.stages import extract_patches,shift_patches,object_patches,exit_waves
from ptycho_jax.DM.operators import (scatter_patches,backproject_patches,object_forward,
    object_adjoint,probe_forward,probe_adjoint,object_diagonal,probe_diagonal)

def relative(x,y):return float(jnp.linalg.norm(x-y)/jnp.maximum(jnp.linalg.norm(y),1e-30))

def test_C3_1_E_dot_and_C3_5_coverage(small_case):
    g,p,o,a=small_case
    patches=extract_patches(o,g)
    y=(patches*.3+.2j)
    adj=scatter_patches(y,g.n,g.canvas_shape)
    assert relative(jnp.vdot(patches,y),jnp.vdot(o,adj))<1e-12
    counts=np.zeros(g.canvas_shape)
    for row,col in np.asarray(g.n):counts[row:row+12,col:col+14]+=1
    np.testing.assert_array_equal(scatter_patches(jnp.ones_like(patches).real,g.n,g.canvas_shape),counts)

def test_C3_2_S_unitary_and_C3_3_L_dot(small_case):
    g,p,o,a=small_case
    patches=extract_patches(o,g)
    shifted=shift_patches(patches,g.delta)
    np.testing.assert_allclose(shift_patches(shifted,-g.delta),patches,rtol=1e-12,atol=2e-15)
    target=jnp.full((4,8,10),.4+.7j)
    assert relative(jnp.vdot(object_patches(o,g),target),jnp.vdot(o,backproject_patches(target,g)))<1e-12


def test_C3_3_C3_4_combined_hermitian_adjoints(small_case):
    g,p,o,a=small_case
    y=exit_waves(p,o,a,g)*(.2+.3j)+.1
    for value,forward,adjoint in (
        (o,lambda v:object_forward(v,p,a,g),lambda v:object_adjoint(v,p,a,g)),
        (p,lambda v:probe_forward(v,o,a,g),lambda v:probe_adjoint(v,o,a,g)),
    ):
        expected=hermitian_adjoint(forward,value)(y)
        np.testing.assert_allclose(adjoint(y),expected,rtol=1e-12,atol=1e-13)
        assert relative(jnp.vdot(forward(value),y),jnp.vdot(value,adjoint(y)))<1e-12


def test_C3_6_integer_object_C3_7_all_probe_normals_diagonal(small_case):
    g,p,o,a=small_case
    for integer in (False,True):
        gg=make_geometry(g.n,np.zeros_like(g.delta),g.canvas_shape,g.frame_shape,g.margin) if integer else g
        np.testing.assert_allclose(probe_adjoint(probe_forward(p,o,a,gg),o,a,gg),probe_diagonal(o,a,gg)*p,rtol=1e-12,atol=1e-13)
        if integer:
            np.testing.assert_allclose(object_adjoint(object_forward(o,p,a,gg),p,a,gg),object_diagonal(p,a,gg)*o,rtol=1e-12,atol=1e-13)


def test_C3_8_fractional_object_diagonal_is_exact_diagonal_not_normal(small_case):
    g,p,o,a=small_case
    diagonal=object_diagonal(p,a,g)
    for row,col in ((1,1),(5,6),(9,9),(18,23)):
        basis=jnp.zeros_like(o).at[row,col].set(1)
        expected=jnp.sum(abs(object_forward(basis,p,a,g))**2)
        np.testing.assert_allclose(diagonal[row,col],expected,rtol=1e-12,atol=1e-13)
    normal=object_adjoint(object_forward(o,p,a,g),p,a,g)
    assert relative(normal,diagonal*o)>1e-3
    assert np.all(np.asarray(diagonal)>=0)


def test_C3_9_C3_10_chunked_adjoint_and_nonnegative_diagonals(small_case):
    g,p,o,a=small_case
    psi=exit_waves(p,o,a,g)
    np.testing.assert_allclose(object_adjoint(psi,p,a,g,chunk_size=3),object_adjoint(psi,p,a,g),rtol=1e-12,atol=1e-13)
    assert np.all(np.asarray(probe_diagonal(o,a,g))>=0)
