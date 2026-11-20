"""Stack model (3D FFT / dft) vs per-exposure model 1: same patterns, timing. python bench_stack.py [cpu|cuda] [x64:0|1]"""
import os, sys, time
plat = sys.argv[1] if len(sys.argv) > 1 else "cpu"
os.environ["JAX_PLATFORMS"] = plat
import jax
jax.config.update("jax_enable_x64", (sys.argv[2] == "1") if len(sys.argv) > 2 else False)
import jax.numpy as jnp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bragg.geometry import BraggGeometry
from bragg.forward import make_bragg_model
from bragg.stack import make_bragg_stack_model, matched_dw
from bragg.reference import grid_axes, sample_phantom

lam, tB, d, Nx, dth, NT, NS = 1e-10, np.deg2rad(20), 30e-9, 64, 1e-4, 11, 32
ct = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
g = BraggGeometry(lam, tB, d, d, matched_dw(tB, lam, Nx, dth))
P, O = (64, 64, 32), (128, 128, 32)
th = (np.arange(NT) - NT // 2) * dth
u, v, _ = grid_axes(P[:2] + (1,), (d, d, g.dw))
f0 = jnp.exp(-(u[:, None] ** 2 + v[None, :] ** 2) / 150e-9**2).astype(ct)
obj = sample_phantom(O, (d, d, g.dw), half_size=300e-9, softness=60e-9).astype(ct)
rng = np.random.default_rng(0)
pos = np.stack([32 + rng.uniform(-6, 6, NS), 32 + rng.uniform(-6, 6, NS), np.zeros(NS)], -1)
def t(f, *a):
    jax.block_until_ready(f(*a)); ts = []
    for _ in range(5):
        s = time.perf_counter(); jax.block_until_ready(f(*a)); ts.append(time.perf_counter() - s)
    return 1e3 * np.median(ts)
res = {}
for mode in ("fft3", "dft"):
    m = make_bragg_stack_model(g, P, th, nx=Nx, mode=mode)
    opt = {"object": obj, "probe_2d": f0, "scan_mistakes": jnp.zeros((NS, 3))}; fx = {"positions": jnp.asarray(pos)}
    idx = jnp.arange(NS)
    fwd = jax.jit(m["batch_model"]); gr = jax.jit(jax.grad(lambda o: m["batch_model"](o, fx, idx).sum()))
    res[f"stack-{mode}"] = (t(fwd, opt, fx, idx), t(gr, opt))
m1 = make_bragg_model(g, P)
opt = {"object": obj, "probe_2d": f0, "scan_mistakes": jnp.zeros((NS * NT, 2))}
fx = {"positions": jnp.asarray(np.repeat(pos, NT, 0)), "thetas": jnp.asarray(np.tile(th, NS))}
idx = jnp.arange(NS * NT)
res["model1 (per exposure)"] = (t(jax.jit(m1["batch_model"]), opt, fx, idx),
                                t(jax.jit(jax.grad(lambda o: m1["batch_model"](o, fx, idx).sum())), opt))
print(f"{plat} x64={jax.config.jax_enable_x64}: {NS} stacks x {NT} angles = {NS*NT} patterns, patch {P}, Nx={Nx}")
for k, (a, b) in res.items(): print(f"  {k:24s} forward {a:8.1f} ms   value+grad {b:8.1f} ms")
