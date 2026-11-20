"""Helpers shared by the Bragg simulation / reconstruction pipelines (no model physics here)."""

import jax
import jax.numpy as jnp
import numpy as np

from .simulate import SimulatedData


def initial_probe_2d(sim: SimulatedData, fwhm_factor=1.25):
    """Deliberately imperfect probe guess: Gaussian with the wrong width (amplitude-normalised)."""
    from .simulate import gaussian_probe_2d

    return gaussian_probe_2d(sim.cfg.n_det, sim.geom.du, sim.cfg.beam_fwhm * fwhm_factor)


def initial_object(shape, pitches, radius=0.3e-6, softness=40e-9):
    """Smooth sphere support (amplitude 1, flat phase) centred in the object array."""
    n = shape
    ax = [(np.arange(k) - k // 2) * p for k, p in zip(n, pitches)]
    R = np.sqrt(ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2 + ax[2][None, None, :] ** 2)
    return (0.5 * (np.tanh((radius - R) / softness) + 1.0)).astype(np.complex64)


def corner_positions(sim: SimulatedData):
    """Nominal patch-corner positions (n_pos, 3) in object voxels (float)."""
    n = sim.cfg.n_det
    corner0 = np.array([sim.obj_shape[0] // 2 - n // 2, sim.obj_shape[1] // 2 - n // 2,
                        sim.obj_shape[2] // 2 - sim.cfg.patch_w // 2], float)
    return corner0 + sim.nominal_vox


def fit_probe_scale(batch_model, opt, fixed, idx, measured_batch):
    """Scale probe_2d so that the mean predicted total counts equal the measured ones."""
    pred = batch_model(opt, fixed, idx)
    ratio = np.asarray(measured_batch).sum() / float(jnp.sum(pred))
    return dict(opt, probe_2d=opt["probe_2d"] * np.sqrt(ratio).astype(np.float32)), ratio


def align_object(rec, truth, max_shift=None):
    """Align rec to truth (envelopes): integer 3D shift by cross-correlation of |.|, then the best
    complex scalar (phase + scale). Returns (aligned, shift, relative error in the support)."""
    a, b = np.abs(rec), np.abs(truth)
    xc = np.fft.ifftn(np.fft.fftn(b) * np.conj(np.fft.fftn(a))).real
    shift = np.array(np.unravel_index(np.argmax(xc), xc.shape))
    shift = np.where(shift > np.array(xc.shape) // 2, shift - np.array(xc.shape), shift)
    if max_shift is not None:
        shift = np.clip(shift, -max_shift, max_shift)
    rec_s = np.roll(rec, tuple(shift), axis=(0, 1, 2))
    mask = np.abs(truth) > 0.2 * np.abs(truth).max()
    c = np.vdot(rec_s[mask], truth[mask]) / np.vdot(rec_s[mask], rec_s[mask])
    aligned = rec_s * c
    err = np.linalg.norm((aligned - truth)[mask]) / np.linalg.norm(truth[mask])
    return aligned, shift, float(err)


def position_error_stats(sim: SimulatedData, mistakes, max_correction, residual_xyz=True):
    """Compare recovered corrections with the true position errors (voxels).

    The true error is true_vox - nominal_vox. A common offset of all positions is
    indistinguishable from an object shift, so the mean is removed from both.
    """
    true_err = sim.true_vox - sim.nominal_vox
    est = np.tanh(np.asarray(mistakes)) * np.asarray(max_correction)
    k = est.shape[1]
    # fitted correction c moves the patch: out(x) = patch(x + c)  ->  c ~ true - nominal... (sign: see docs)
    te = true_err[:, :k]
    te0 = te - te.mean(0)
    out = {}
    for sign in (+1, -1):
        e0 = sign * est - (sign * est).mean(0)
        out[sign] = np.sqrt(np.mean((e0 - te0) ** 2, axis=0))
    return true_err, est, out, np.sqrt(np.mean(te0**2, axis=0))


def tie_model1_corrections(model, n_rock):
    """Model 1 evaluates single (position, angle) exposures. All angles of one scan position share
    one position error, so tie them: `scan_mistakes` has one row per scan position (n_pos, 2) and the
    exposure index i maps to position i // n_rock. Returns a model dict with the same interface.
    Exposures must be ordered position-major (all angles of position 0, then position 1, ...)."""

    def view(opt, i):
        return dict(opt, scan_mistakes=jnp.repeat(opt["scan_mistakes"], n_rock, axis=0))

    def point_model(opt, fixed, i):
        return model["point_model"](view(opt, i), fixed, i)

    def batch_model(opt, fixed, idx):
        return model["batch_model"](view(opt, None), fixed, idx)

    return dict(model, point_model=point_model, batch_model=batch_model)


def show_object_slices(obj, title="", mask_thr=0.2):
    import matplotlib.pyplot as plt

    obj = np.asarray(obj)
    n = obj.shape
    a = np.abs(obj)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    views = [("(u,v) @ w mid", lambda x: x[:, :, n[2] // 2]),
             ("(u,w) @ v mid", lambda x: x[:, n[1] // 2, :]),
             ("(v,w) @ u mid", lambda x: x[n[0] // 2, :, :])]
    for j, (name, f) in enumerate(views):
        ax[0, j].imshow(f(a), cmap="viridis", vmin=0, vmax=max(a.max(), 1e-9))
        ax[0, j].set_title(f"|obj| {name}")
        ph = np.angle(f(obj)) * (f(a) > mask_thr * a.max())
        ax[1, j].imshow(ph, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[1, j].set_title("phase (masked)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def thickness_support(obj_shape, pitches, thickness, lateral_halfwidth=None):
    """Known-extent support on the beam grid (u,v,w), centred in the array.

    thickness: full extent along the beam axis w [m] (the sample thickness seen by the beam).
    lateral_halfwidth: optional half-width [m] of a (u,v) box. Returns a float mask (Nu,Nv,Nw).
    """
    ax = [(np.arange(k) - k // 2) * p for k, p in zip(obj_shape, pitches)]
    m = (np.abs(ax[2]) <= thickness / 2)[None, None, :] * np.ones(obj_shape, bool)
    if lateral_halfwidth is not None:
        m &= (np.abs(ax[0]) <= lateral_halfwidth)[:, None, None]
        m &= (np.abs(ax[1]) <= lateral_halfwidth)[None, :, None]
    return m.astype(np.float32)


def make_projection(support=None, frozen=None):
    """projection_fn(params, fixed) for `run_optimization_loop`: multiplies the object by the support
    after each update and (optionally) restores frozen parameters to given values."""
    support = None if support is None else jnp.asarray(support)
    frozen = frozen or {}

    def projection(params, fixed):
        params = dict(params)
        if support is not None:
            params["object"] = params["object"] * support
        for k, v in frozen.items():
            params[k] = v
        return params, fixed

    return projection
