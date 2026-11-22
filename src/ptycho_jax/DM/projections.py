"""Modulus constraint and alternating analytical overlap block solves."""

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from .conventions import fft2c, ifft2c
from .stages import detector_intensity, object_patches, exit_waves
from .operators import object_adjoint, object_diagonal, backproject_patches
from .gauge import normalize_all, active_modes


def fourier_projection(psi, sqrt_I, mask, eps):
    """Project (J,K,Y,X) waves onto masked amplitudes (J,Y,X); return stats.

    At a zero predicted amplitude, seed the first incoherent channel rather
    than multiplying zero by infinity. This is one deterministic choice among
    the nonunique modulus projections at zero, not a noise-relaxed constraint.
    """
    mask = jnp.broadcast_to(mask, sqrt_I.shape)
    f = fft2c(psi)
    amplitude = jnp.sqrt(detector_intensity(f))
    good = amplitude > eps
    denominator = jnp.where(good, amplitude, jnp.ones_like(amplitude))
    projected = f * (sqrt_I / denominator)[:, None]
    seed = jnp.zeros_like(f).at[:, 0].set(sqrt_I.astype(f.dtype))
    projected = jnp.where(good[:, None], projected, seed)
    projected = jnp.where(mask[:, None], projected, f)
    energy = jnp.maximum(
        jnp.sum(jnp.where(mask, sqrt_I**2, 0)), jnp.finfo(sqrt_I.dtype).tiny
    )
    error = jnp.sum(jnp.where(mask, (amplitude - sqrt_I) ** 2, 0)) / energy
    return ifft2c(projected), {"fourier_error": error}


def solve_object(psi, probe, obj_old, a, geom, cfg):
    """Damped exact object LS: (J,K,Y,X) target -> (H,W), scalar alpha.

    Integer positions have diagonal normals; fractional positions use exact
    matrix-free CG, preconditioned by the exact normal diagonal.
    """
    diagonal = object_diagonal(probe, a, geom)
    alpha = jnp.maximum(
        cfg.alpha_object * jnp.max(diagonal), jnp.finfo(obj_old.real.dtype).eps
    )
    rhs = object_adjoint(psi, probe, a, geom, cfg.chunk_size) + alpha * obj_old
    if geom.integer_only:
        return rhs / (diagonal + alpha), alpha
    illumination = jnp.einsum(
        "jk,kyx->jyx", a * a, abs(probe) ** 2, precision=jax.lax.Precision.HIGHEST
    )

    def normal(obj):
        return (
            backproject_patches(
                illumination * object_patches(obj, geom), geom, cfg.chunk_size
            )
            + alpha * obj
        )

    solution, _ = cg(
        normal,
        rhs,
        x0=obj_old,
        M=lambda v: v / (diagonal + alpha),
        tol=cfg.object_cg_tolerance,
        maxiter=cfg.object_cg_iterations,
    )
    return solution, alpha


def solve_probe(psi, probe_old, obj, a, geom, cfg):
    """Exact damped probe LS: target (J,K,Y,X) -> (K,Y,X), alpha (K,)."""
    patches = object_patches(obj, geom)
    diagonal = jnp.einsum(
        "jk,jyx->kyx", a * a, abs(patches) ** 2, precision=jax.lax.Precision.HIGHEST
    )
    alpha = jnp.maximum(
        cfg.alpha_probe * jnp.max(diagonal, axis=(-2, -1)),
        jnp.finfo(probe_old.real.dtype).eps,
    )
    numerator = jnp.sum(a[:, :, None, None] * jnp.conj(patches[:, None]) * psi, axis=0)
    result = (numerator + alpha[:, None, None] * probe_old) / (
        diagonal + alpha[:, None, None]
    )
    active = active_modes(a, cfg.collapse_threshold)
    return jnp.where(active[:, None, None], result, probe_old), alpha


def solve_weights(
    psi, probe, obj, a_old, geom, weight_mode, eps_relative=1e-8, blend=1.0
):
    """Nonnegative amplitude LS (J,K); mode 0=fixed, 1=global, 2=per-shot."""
    u = probe[None] * object_patches(obj, geom)[:, None]
    numerator = jnp.sum(jnp.real(jnp.conj(u) * psi), axis=(-2, -1))
    denominator = jnp.sum(abs(u) ** 2, axis=(-2, -1))
    epsilon = eps_relative * jnp.mean(denominator)
    safe = lambda x: jnp.maximum(x, jnp.finfo(a_old.dtype).tiny)
    per_shot = jnp.maximum(numerator, 0) / safe(denominator + epsilon)
    global_profile = jnp.maximum(jnp.sum(numerator, axis=0), 0) / safe(
        jnp.sum(denominator, axis=0) + epsilon
    )
    global_a = jnp.broadcast_to(global_profile, a_old.shape)
    return jax.lax.switch(
        weight_mode,
        (
            lambda: a_old,
            lambda: global_a,
            lambda: (1 - blend) * global_a + blend * per_shot,
        ),
    )


def overlap_projection(
    psi, probe, obj, a, geom, cfg, update_probe, weight_mode, covered
):
    """Warm alternating fit to (J,K,Y,X); return waves, probe, object, a.

    Dynamic gates are scalar JAX values. Gauge operations are skipped when a
    corresponding parameter block is frozen, preserving bit-identical gates.
    """

    def inner(_, carry):
        p, o, aa = carry
        o, _ = solve_object(psi, p, o, aa, geom, cfg)
        p = jax.lax.cond(
            update_probe,
            lambda p: solve_probe(psi, p, o, aa, geom, cfg)[0],
            lambda p: p,
            p,
        )
        aa = jax.lax.cond(
            weight_mode > 0,
            lambda aa: solve_weights(
                psi,
                p,
                o,
                aa,
                geom,
                weight_mode,
                cfg.weight_eps_relative,
                cfg.weight_blend,
            ),
            lambda aa: aa,
            aa,
        )
        return jax.lax.cond(
            update_probe & (weight_mode > 0),
            lambda vals: normalize_all(*vals, covered),
            lambda vals: vals,
            (p, o, aa),
        )

    probe, obj, a = jax.lax.fori_loop(0, cfg.n_inner, inner, (probe, obj, a))
    return exit_waves(probe, obj, a, geom), probe, obj, a
