"""float32 vs float64 behaviour of the near-field (angular spectrum) probe propagator.

float32 needs jax_enable_x64=False at start-up, so each precision runs in a fresh subprocess.
Run with `-s` to see the printed table.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent(
    """
    import sys, json
    sys.path.insert(0, {root!r})
    import jax
    jax.config.update("jax_enable_x64", {x64})
    import jax.numpy as jnp, numpy as np
    from bragg.probe import angular_spectrum_stack

    lam, d, N = 1.3776e-10, 38.7e-9, 64
    x = (np.arange(N) - N // 2) * d
    w0 = 400e-9
    f0 = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / w0**2)
    z = (np.arange(33) - 16) * d * 4.0
    dt = jnp.complex128 if {x64} else jnp.complex64
    out = angular_spectrum_stack(jnp.asarray(f0, dtype=dt), lam, d, d, z)
    naive_fn = None
    # naive formula for comparison (cancellation prone)
    fu = jnp.fft.fftfreq(N, d=d)[:, None]
    fv = jnp.fft.fftfreq(N, d=d)[None, :]
    kz = jnp.sqrt(jnp.clip(1.0 / lam**2 - fu**2 - fv**2, 0.0))
    H = jnp.exp(2j * jnp.pi * (kz - 1.0 / lam)[..., None] * jnp.asarray(z)[None, None, :])
    naive = jnp.fft.ifft2(jnp.fft.fft2(jnp.asarray(f0, dtype=dt))[..., None] * H, axes=(0, 1))
    np.save({path!r}, np.stack([np.asarray(out), np.asarray(naive)]))
    print(json.dumps({{"dtype": str(out.dtype)}}))
    """
)


def run(x64, tmp_path):
    path = str(tmp_path / f"out_{int(x64)}.npy")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    r = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(root=ROOT, x64=x64, path=path)],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, r.stderr
    return np.load(path), json.loads(r.stdout.strip().splitlines()[-1])["dtype"]


def test_propagator_precision(tmp_path):
    (stable64, naive64), dt64 = run(True, tmp_path)
    (stable32, naive32), dt32 = run(False, tmp_path)
    assert dt64 == "complex128" and dt32 == "complex64"
    scale = np.abs(stable64).max()
    e_stable = np.abs(stable32 - stable64).max() / scale
    e_naive = np.abs(naive32 - stable64).max() / scale
    e_naive64 = np.abs(naive64 - stable64).max() / scale
    print(f"\n  float64: naive vs stable formula     : {e_naive64:.2e}")
    print(f"  float32: stable formula vs float64   : {e_stable:.2e}")
    print(f"  float32: naive formula  vs float64   : {e_naive:.2e}")
    assert e_naive64 < 1e-6
    assert e_stable < 1e-4  # float32 is fine with the stable formula
    assert e_naive > 10 * e_stable  # the naive formula is clearly worse in float32


def test_rocking_dQ_and_forward_float32(tmp_path):
    """dQ(theta) and the full forward model in float32 vs float64 (uses bench_precision worker)."""
    sys.path.insert(0, ROOT)
    import bench_precision as bp

    r64, p64 = bp.run(True, "cpu")
    r32, p32 = bp.run(False, "cpu")
    I64, I32 = np.load(p64), np.load(p32)
    err = np.abs(I32 - I64).max(axis=(1, 2)) / I64.max()
    print(f"\n  dQ abs error f32: {r32['dQ_abs_err_1_per_m']:.2e} 1/m (max dQ {r32['dQ_max_1_per_m']:.1e})")
    print(f"  forward f32 vs f64, worst over rocking angles: {err.max():.2e}")
    assert r32["dQ_abs_err_1_per_m"] < 5.0  # was ~177 1/m with the subtractive formula
    assert r64["dQ_abs_err_1_per_m"] < 1e-5
    assert err.max() < 1e-5
