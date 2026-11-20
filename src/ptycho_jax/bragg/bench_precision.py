"""Precision + performance of the Bragg forward model in float32 vs float64.

Usage: python bench_precision.py [cpu|cuda]   (each precision runs in a subprocess,
because jax_enable_x64 must be set before JAX starts). Writes results to stdout.
"""
import json, os, subprocess, sys, textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKER = textwrap.dedent("""
    import sys, json, time
    sys.path.insert(0, {root!r})
    import jax
    jax.config.update("jax_enable_x64", {x64})
    import jax.numpy as jnp, numpy as np
    from bragg.geometry import BraggGeometry, delta_Q_beam_frame
    from bragg.forward import make_bragg_model, fourier_shift_uv, detector_amplitude
    from bragg.probe import build_reference_probe, rotate_volume_fourier_shear
    from bragg.reference import grid_axes, sample_phantom

    ctype = jnp.complex128 if {x64} else jnp.complex64
    lam = 12.398419843320026e-10 / 9.0
    tB = np.arcsin(lam / (2 * 10.12e-10 / 4)); d = 38.7e-9
    geom = BraggGeometry(lam, tB, d, d, d)
    PATCH, OBJ = (64, 64, 32), (128, 128, 32)
    pitches = (d, d, d)
    thetas = np.array([0, 1e-6, 1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 1.5e-3, -5e-4])
    out = {{}}

    # --- dQ itself: compare against float64 numpy reference
    q0 = 2 * np.sin(tB) / lam
    dq_true = np.stack([q0 * np.array([np.sin(t), 0.0, -2 * np.sin(t / 2) ** 2]) for t in thetas])
    B = np.array([[np.sin(tB), 0, -np.cos(tB)], [0, 1, 0], [np.cos(tB), 0, np.sin(tB)]])
    dq_true = dq_true @ B.T
    dq = np.stack([np.asarray(delta_Q_beam_frame(jnp.asarray(t), tB, lam)) for t in thetas])
    out["dQ_abs_err_1_per_m"] = float(np.abs(dq - dq_true).max())
    out["dQ_max_1_per_m"] = float(np.abs(dq_true).max())
    out["dQ_phase_err_rad_at_half_patch"] = float(2 * np.pi * np.abs(dq - dq_true).max() * 32 * d)

    # --- full model, all thetas
    u, v, _ = grid_axes(PATCH[:2] + (1,), pitches)
    f0 = jnp.exp(-(u[:, None] ** 2 + v[None, :] ** 2) / 400e-9**2).astype(ctype)
    model = make_bragg_model(geom, PATCH, crop=4)
    obj = sample_phantom(OBJ, pitches, half_size=150e-9, softness=40e-9).astype(ctype)
    n = len(thetas)
    opt = {{"object": obj, "probe_2d": f0, "scan_mistakes": jnp.zeros((n, 2))}}
    fx = {{"positions": jnp.asarray(np.tile([32.3, 31.7, 0.0], (n, 1))), "thetas": jnp.asarray(thetas)}}
    I = np.asarray(jax.jit(model["batch_model"])(opt, fx, jnp.arange(n)), dtype=np.float64)
    np.save({path!r}, I)
    out["dtype"] = str(jnp.asarray(I).dtype); out["device"] = str(jax.devices()[0])

    # --- timings (median of 5, after warm-up), batch of 256 positions
    N = 256
    opt2 = {{"object": obj, "probe_2d": f0, "scan_mistakes": jnp.zeros((N, 2))}}
    rng = np.random.default_rng(0)
    fx2 = {{"positions": jnp.asarray(np.stack([32 + rng.uniform(-6, 6, N), 32 + rng.uniform(-6, 6, N), np.zeros(N)], -1)),
           "thetas": jnp.asarray(rng.uniform(-1e-3, 1e-3, N))}}
    def timeit(f, *a):
        r = f(*a); jax.block_until_ready(r)
        ts = []
        for _ in range(5):
            t = time.perf_counter(); jax.block_until_ready(f(*a)); ts.append(time.perf_counter() - t)
        return 1e3 * float(np.median(ts))
    bm = jax.jit(model["batch_model"]); idx = jnp.arange(N)
    out["ms_forward_batch256"] = timeit(bm, opt2, fx2, idx)
    gr = jax.jit(jax.grad(lambda o: model["batch_model"](o, fx2, idx).sum()))
    out["ms_grad_batch256"] = timeit(gr, opt2)
    out["ms_probe_build"] = timeit(jax.jit(lambda o: model["probe_constructor"](o)), opt2)
    vol = jnp.ones((64, 64, 32), ctype)
    out["ms_fft2_patch_x256"] = timeit(jax.jit(lambda x: jnp.fft.fft2(x[None] * jnp.ones((256, 1, 1, 1), ctype), axes=(1, 2))), vol)
    print("RESULT" + json.dumps(out))
""")

def run(x64, platform):
    path = f"/tmp/bench_I_{int(x64)}.npy"
    env = dict(os.environ, JAX_PLATFORMS=platform)
    r = subprocess.run([sys.executable, "-c", WORKER.format(root=ROOT, x64=x64, path=path)],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-2000:]); sys.exit(1)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")][0]
    return json.loads(line[6:]), path

if __name__ == "__main__":
    import numpy as np
    platform = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    r64, p64 = run(True, platform); r32, p32 = run(False, platform)
    I64, I32 = np.load(p64), np.load(p32)
    err = np.abs(I32 - I64).max(axis=(1, 2)) / I64.max()
    print(f"platform {platform}: {r64['device']}")
    print("rel. max error float32 vs float64 per rocking angle:", np.array2string(err, precision=2))
    for k in r64:
        if k.startswith(("dQ", "ms_")):
            print(f"{k:36s} f64={r64[k]:.4g}   f32={r32[k]:.4g}")
