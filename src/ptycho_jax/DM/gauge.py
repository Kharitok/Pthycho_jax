"""Exact compensated scale/phase operations; no per-shot mode rotation."""

import jax.numpy as jnp


def normalize_modes(probe, a):
    """Probe (K,Y,X), amplitudes (J,K) -> unit modes with compensated amplitudes."""
    norms = jnp.sqrt(jnp.sum(abs(probe) ** 2, axis=(-2, -1)))
    norms = jnp.where(norms > 0, norms, jnp.ones_like(norms))
    return probe / norms[:, None, None], a * norms[None]


def fix_object_gauge(probe, obj, a, covered):
    """Fix mean object amplitude/phase over covered (H,W), preserving (J,K,Y,X) fields."""
    count = jnp.maximum(jnp.sum(covered), 1)
    scale = jnp.sum(jnp.where(covered, abs(obj), 0)) / count
    scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    phase = jnp.angle(jnp.sum(jnp.where(covered, obj, 0)))
    rotation = jnp.exp(1j * phase).astype(probe.dtype)
    return probe * rotation, obj / (scale * rotation), a * scale


def normalize_all(probe, obj, a, covered):
    """Compensated normalization of (K,Y,X), (H,W), (J,K)."""
    probe, a = normalize_modes(probe, a)
    return fix_object_gauge(probe, obj, a, covered)


def active_modes(a, threshold=1e-3):
    """Boolean active mask (K,) from per-mode maximum amplitudes (J,K)."""
    maxima = jnp.max(a, axis=0)
    return (maxima > 0) & (maxima >= threshold * jnp.max(maxima))
