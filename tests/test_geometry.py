import jax
import jax.numpy as jnp
import numpy as np

from bragg.geometry import (
    beam_basis_matrix,
    bragg_Q_vector,
    delta_Q_beam_frame,
    rotation_matrix_y,
)


def test_rotation_identity_and_orthogonal():
    assert jnp.allclose(rotation_matrix_y(0.0), jnp.eye(3))
    R = rotation_matrix_y(0.3)
    assert jnp.allclose(R @ R.T, jnp.eye(3))


def test_beam_basis_orthonormal_right_handed(geom):
    B = beam_basis_matrix(geom.theta_B)
    assert jnp.allclose(B @ B.T, jnp.eye(3))
    assert jnp.allclose(jnp.cross(B[0], B[1]), B[2])


def test_Q0_and_kf_ki(geom):
    Q0 = bragg_Q_vector(0.0, geom.theta_B, geom.wavelength)
    assert jnp.allclose(Q0[:2], 0.0)
    assert jnp.isclose(Q0[2], 2 * np.sin(geom.theta_B) / geom.wavelength)
    B = beam_basis_matrix(geom.theta_B)
    # in the beam frame Q0 has no v component and kf - ki = Q0 with kf || w
    Q0b = B @ Q0
    kf = jnp.array([0.0, 0.0, 1.0 / geom.wavelength])
    ki = kf - Q0b
    assert jnp.isclose(jnp.linalg.norm(ki), 1 / geom.wavelength)
    assert jnp.isclose(ki[0], np.sin(2 * geom.theta_B) / geom.wavelength)


def test_dQ_zero_at_reference_and_norm_preserved(geom):
    assert jnp.allclose(delta_Q_beam_frame(0.0, geom.theta_B, geom.wavelength), 0.0)
    th = 0.01
    Q = bragg_Q_vector(th, geom.theta_B, geom.wavelength)
    assert jnp.isclose(jnp.linalg.norm(Q), geom.q_mag)


def test_dQ_first_order(geom):
    J = jax.jacfwd(delta_Q_beam_frame)(0.0, geom.theta_B, geom.wavelength)
    lam, tB = geom.wavelength, geom.theta_B
    expected = jnp.array([2 * np.sin(tB) ** 2, 0.0, np.sin(2 * tB)]) / lam
    assert jnp.allclose(J, expected, rtol=1e-10)
