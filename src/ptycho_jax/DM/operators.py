"""Complex-linear operators and exact Hermitian adjoints for fixed positions."""

from dataclasses import replace
import jax
import jax.numpy as jnp
from .stages import object_patches, shift_patches, shift_ramps, taper


def scatter_patches(patches, origins, canvas_shape, *, method="scatter"):
    """Scatter-add (J,Y,X) at integer origins (J,2) -> canvas (H,W).

    Index arrays are broadcast on demand, never stored in Geometry. A scan
    implementation is available for independent checks and memory benchmarks.
    """
    canvas = jnp.zeros(canvas_shape, dtype=patches.dtype)
    if method == "scan":

        def add(carry, items):
            patch, origin = items
            old = jax.lax.dynamic_slice(carry, origin, patch.shape)
            return jax.lax.dynamic_update_slice(carry, old + patch, origin), None

        return jax.lax.scan(add, canvas, (patches, origins))[0]
    rows = origins[:, 0, None, None] + jnp.arange(patches.shape[-2])[None, :, None]
    cols = origins[:, 1, None, None] + jnp.arange(patches.shape[-1])[None, None, :]
    return canvas.at[rows, cols].add(patches)


def _backproject_direct(patches, geom):
    m = geom.margin
    if geom.integer_only:
        return scatter_patches(patches, geom.n + m, geom.canvas_shape)
    padded = jnp.pad(patches, ((0, 0), (m, m), (m, m)))
    back = shift_patches(padded, geom.delta)
    back = back * taper(back.shape[-2:], m, patches.real.dtype)
    return scatter_patches(back, geom.n, geom.canvas_shape)


def backproject_patches(patches, geom, chunk_size=0):
    """Hermitian L*: (J,Y,X) -> canvas; optionally sum bounded shot chunks."""
    if chunk_size <= 0 or chunk_size >= len(geom.n):
        return _backproject_direct(patches, geom)
    j = len(geom.n)
    extra = (-j) % chunk_size
    values = jnp.pad(patches, ((0, extra), (0, 0), (0, 0)))
    origins = jnp.pad(geom.n, ((0, extra), (0, 0)))
    fractions = jnp.pad(geom.delta, ((0, extra), (0, 0)))

    def add(canvas, index):
        start = index * chunk_size
        p = jax.lax.dynamic_slice_in_dim(values, start, chunk_size)
        n = jax.lax.dynamic_slice_in_dim(origins, start, chunk_size)
        delta = jax.lax.dynamic_slice_in_dim(fractions, start, chunk_size)
        g = replace(geom, n=n, delta=delta)
        return canvas + _backproject_direct(p, g), None

    return jax.lax.scan(
        add,
        jnp.zeros(geom.canvas_shape, patches.dtype),
        jnp.arange((j + extra) // chunk_size),
    )[0]


def object_forward(obj, probe, a, geom):
    """A_O: complex canvas (H,W) -> exit waves (J,K,Y,X)."""
    return a[:, :, None, None] * probe[None] * object_patches(obj, geom)[:, None]


def object_adjoint(psi, probe, a, geom, chunk_size=0):
    """A_O*: complex waves (J,K,Y,X) -> canvas (H,W)."""
    patches = jnp.sum(a[:, :, None, None] * jnp.conj(probe[None]) * psi, axis=1)
    return backproject_patches(patches, geom, chunk_size)


def probe_forward(probe, obj, a, geom):
    """A_P: probe (K,Y,X) -> exit waves (J,K,Y,X), at fixed object/weights."""
    return object_forward(obj, probe, a, geom)


def probe_adjoint(psi, obj, a, geom):
    """A_P*: exit waves (J,K,Y,X) -> probe (K,Y,X)."""
    patches = object_patches(obj, geom)
    return jnp.sum(a[:, :, None, None] * jnp.conj(patches[:, None]) * psi, axis=0)


def probe_diagonal(obj, a, geom):
    """Exact normal diagonal D_P (K,Y,X), also for fractional positions."""
    return jnp.einsum(
        "jk,jyx->kyx",
        a * a,
        abs(object_patches(obj, geom)) ** 2,
        precision=jax.lax.Precision.HIGHEST,
    )


def object_diagonal(probe, a, geom):
    """Exact normal diagonal diag(A_O* A_O), canvas (H,W), nonnegative.

    Fractional shifts make the full normal non-diagonal. Its diagonal is a
    circular correlation with the squared shift impulse response. It is used
    as a CG preconditioner, never as an exact fractional normal operator.
    """
    illumination = jnp.einsum(
        "jk,kyx->jyx", a * a, abs(probe) ** 2, precision=jax.lax.Precision.HIGHEST
    )
    m = geom.margin
    if geom.integer_only:
        return scatter_patches(illumination, geom.n + m, geom.canvas_shape)
    padded = jnp.pad(illumination, ((0, 0), (m, m), (m, m)))
    ramp = shift_ramps(-geom.delta, padded.shape[-2:], probe.real.dtype)
    impulse = jnp.fft.ifft2(ramp, axes=(-2, -1))
    kernel_power = abs(impulse) ** 2
    correlation = jnp.fft.ifft2(
        jnp.fft.fft2(padded) * jnp.conj(jnp.fft.fft2(kernel_power))
    ).real
    diagonal = (
        jnp.maximum(correlation, 0) * taper(padded.shape[-2:], m, probe.real.dtype) ** 2
    )
    return jnp.maximum(scatter_patches(diagonal, geom.n, geom.canvas_shape), 0)


def coverage_map(geom, dtype=jnp.float32):
    """Nominal central-frame coverage (H,W), independent of object and probe."""
    return scatter_patches(
        jnp.ones((len(geom.n), *geom.frame_shape), dtype),
        geom.n + geom.margin,
        geom.canvas_shape,
    )
