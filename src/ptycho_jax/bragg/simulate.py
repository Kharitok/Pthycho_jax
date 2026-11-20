"""Data simulation for 3D Bragg ptychography by a DIRECT 3D FFT (independent of the models).

What is independent of the reconstruction models (`forward.py`, `stack.py`):
  * the probe is the analytic Gaussian beam evaluated point by point on the beam grid
    (no angular-spectrum propagation, no Fourier-shear rotation);
  * the particle is an analytic lab-frame function evaluated at the exact (fractional)
    position and PHYSICALLY ROTATED by theta (no phase-ramp approximation of the rocking
    envelope, no Fourier shifts);
  * intensities come from a full `fftn` of probe*object*exp(2 pi i dQ.r) over the whole
    volume; the detector plane is the q_w = 0 plane of that 3D transform
    (the Ewald-sphere curvature is < 0.1 q_w plane for the geometries used here).
Shared with the models (necessarily): conventions of `geometry.py` (frames, dQ, beam basis)
and the voxel pitch du = lambda L / (N p) fixed by the detector.

Positions are defined as in the models: patch (probe) centre relative to the object centre
in the beam frame (u, v, w), in metres; nominal values come from `scan_to_grid_positions`.
Convention "focus tracking": the probe focus sits at the patch centre (w = 0), exactly the
convention the models assume.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import (
    BraggGeometry,
    beam_basis_matrix,
    delta_Q_beam_frame,
    rotation_matrix_y,
    scan_to_grid_positions,
)
from .reference import gaussian_beam_envelope, grid_axes

HC_KEV_ANGSTROM = 12.398419843320026


@dataclass
class ExperimentConfig:
    energy_kev: float = 9.0
    pixel: float = 55e-6
    det_dist: float = 1.35
    n_det: int = 256  # detector window (n_det x n_det)
    d_hkl: float = 2.915e-10  # beta-Sn (200)
    dw_factor: float = 1.0  # dw = dw_factor * du
    patch_w: int = 64  # voxels along w of the probe volume / patch
    obj_margin: int = 48  # extra object voxels (u,v) around the patch for scan moves
    n_rock: int = 11
    rock_step: Optional[float] = None  # rad; None -> matched to the 3D-FFT lattice
    nx: Optional[int] = None  # sheared slots for the stack model; None -> auto
    beam_fwhm: float = 300e-9
    n_scan_x: int = 7
    n_scan_y: int = 5
    step_x: float = 0.6e-6  # m along sample x (-> u shift of only step*sin(tB)/cos(2tB))
    step_y: float = 0.2e-6
    pos_err_uv_vox: float = 1.5  # std of the true position errors (voxels)
    pos_err_w_vox: float = 1.0
    photons_max: Optional[float] = 1e6  # peak-pattern total photons; None -> noise free
    seed: int = 0

    def derived(self):
        lam = HC_KEV_ANGSTROM * 1e-10 / self.energy_kev
        theta_B = float(np.arcsin(lam / (2 * self.d_hkl)))
        du = lam * self.det_dist / (self.n_det * self.pixel)
        dw = self.dw_factor * du
        geom = BraggGeometry(lam, theta_B, du, du, dw)
        return geom


def make_particle(size=0.5e-6, strain_rms=0.6, amp_var=0.12, softness=25e-9, seed=1,
                  aspect=(0.9, 1.0, 1.1), n_modes=10):
    """Analytic lab-frame particle fn(x, y, z) -> complex envelope (Bragg carrier removed).

    Rounded box (superellipsoid-like, soft edges) of ~`size`; amplitude 1 +- `amp_var`
    from a smooth random field; phase = smooth random 'strain' field (rms `strain_rms`)
    plus a weak linear gradient. All fields are sums of a few low-order cosines.
    """
    rng = np.random.default_rng(seed)
    half = 0.5 * size * np.asarray(aspect)
    L = size
    K = rng.integers(1, 3, size=(2, n_modes, 3)) * rng.choice([-1, 1], size=(2, n_modes, 3))
    ph = rng.uniform(0, 2 * np.pi, size=(2, n_modes))
    wt = rng.normal(size=(2, n_modes))
    wt /= np.sqrt((wt**2).sum(axis=1, keepdims=True) / 2)  # unit variance fields
    grad = rng.normal(size=3) * 0.5 / L

    K, ph, wt, half, grad = (jnp.asarray(a) for a in (K, ph, wt, half, grad))

    def field_(i, x, y, z):
        arg = 2 * jnp.pi * (K[i, :, 0] * x[..., None] + K[i, :, 1] * y[..., None]
                            + K[i, :, 2] * z[..., None]) / (2 * L) + ph[i]
        return jnp.sum(wt[i] * jnp.cos(arg), axis=-1)

    def edge(t, h):
        return 0.5 * (jnp.tanh((h - jnp.abs(t)) / softness) + 1.0)

    def fn(x, y, z):
        shape = edge(x, half[0]) * edge(y, half[1]) * edge(z, half[2])
        amp = shape * (1.0 + amp_var * field_(0, x, y, z))
        phase = strain_rms * field_(1, x, y, z) + 2 * jnp.pi * (grad[0] * x + grad[1] * y + grad[2] * z)
        return amp * jnp.exp(1j * phase)

    return fn


def gaussian_probe_2d(n, pitch, fwhm):
    """2D complex Gaussian field at the focus (amplitude exp(-r^2/w0^2), intensity FWHM = fwhm)."""
    w0 = fwhm / np.sqrt(2 * np.log(2))
    x = (np.arange(n) - n // 2) * pitch
    return np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / w0**2).astype(complex)


def nominal_positions(cfg: ExperimentConfig, geom: BraggGeometry):
    """Scan grid on the sample (X, Y in m) and the patch-centre offsets in voxels (u, v, w=0)."""
    X = (np.arange(cfg.n_scan_x) - (cfg.n_scan_x - 1) / 2) * cfg.step_x
    Y = (np.arange(cfg.n_scan_y) - (cfg.n_scan_y - 1) / 2) * cfg.step_y
    XX, YY = [a.ravel() for a in np.meshgrid(X, Y, indexing="ij")]
    vox = np.asarray(scan_to_grid_positions(XX, YY, geom.theta_B, geom.du, geom.dv, geom.dw))
    return XX, YY, vox  # vox (n_pos, 3): centre offset in voxels relative to the object centre


def rocking_angles(cfg: ExperimentConfig, geom: BraggGeometry, nx: int):
    from .stack import matched_rocking_step

    step = cfg.rock_step if cfg.rock_step is not None else matched_rocking_step(geom, nx)
    return (np.arange(cfg.n_rock) - (cfg.n_rock - 1) / 2) * step, step


def default_nx(cfg: ExperimentConfig, geom: BraggGeometry):
    from .stack import min_nx

    n = min_nx(geom, cfg.n_det, cfg.patch_w)
    return int(cfg.nx) if cfg.nx is not None else int(np.ceil(n / 16) * 16)


@dataclass
class SimulatedData:
    cfg: ExperimentConfig
    geom: BraggGeometry
    thetas: np.ndarray  # (n_rock,)
    scan_xy: np.ndarray  # (n_pos, 2) nominal sample-plane scan (m)
    nominal_vox: np.ndarray  # (n_pos, 3) nominal patch-centre offsets (voxels)
    true_vox: np.ndarray  # (n_pos, 3) true patch-centre offsets (voxels)
    intensities: np.ndarray  # (n_pos, n_rock, n_det, n_det) (noisy if photons_max)
    clean: np.ndarray  # noise-free intensities (same scale)
    scale: float  # photons per unit model intensity
    probe_2d: np.ndarray  # true focus field (n_det, n_det)
    true_object: np.ndarray  # (n_obj_u, n_obj_v, patch_w) envelope on the beam grid
    obj_shape: Tuple[int, int, int]
    nx: int
    extra: dict = field(default_factory=dict)

    def save(self, path):
        d = dict(
            thetas=self.thetas, scan_xy=self.scan_xy, nominal_vox=self.nominal_vox,
            true_vox=self.true_vox, intensities=self.intensities, clean=self.clean,
            scale=self.scale, probe_2d=self.probe_2d, true_object=self.true_object,
            nx=self.nx,
        )
        for k, v in vars(self.cfg).items():
            d[f"cfg_{k}"] = np.asarray(-1 if v is None else v)
        np.savez_compressed(path, **d)

    @staticmethod
    def load(path):
        z = np.load(path)
        cfg = ExperimentConfig(**{
            k[4:]: (None if float(z[k]) == -1 and k[4:] in ("rock_step", "nx", "photons_max") else z[k].item())
            for k in z.files if k.startswith("cfg_")
        })
        for f_ in ("n_det", "patch_w", "obj_margin", "n_rock", "n_scan_x", "n_scan_y", "seed"):
            setattr(cfg, f_, int(getattr(cfg, f_)))
        geom = cfg.derived()
        obj = z["true_object"]
        return SimulatedData(cfg, geom, z["thetas"], z["scan_xy"], z["nominal_vox"], z["true_vox"],
                             z["intensities"], z["clean"], float(z["scale"]), z["probe_2d"], obj,
                             obj.shape, int(z["nx"]))


def _sim_grid(n_uv, n_w, geom):
    u, v, w = grid_axes((n_uv, n_uv, n_w), (geom.du, geom.dv, geom.dw))
    return jnp.meshgrid(u, v, w, indexing="ij")


def make_probe_volume(cfg: ExperimentConfig, geom: BraggGeometry):
    """Analytic probe envelope on the beam grid (n_det, n_det, patch_w), focus at the centre."""
    U, V, W = _sim_grid(cfg.n_det, cfg.patch_w, geom)
    s2, c2 = np.sin(2 * geom.theta_B), np.cos(2 * geom.theta_B)
    u_p = U * c2 - W * s2  # lateral coordinate w.r.t. the incident beam
    w_p = U * s2 + W * c2  # along the incident beam
    w0 = cfg.beam_fwhm / np.sqrt(2 * np.log(2))
    return gaussian_beam_envelope(u_p, V, w_p, w0, geom.wavelength)


def simulate_dataset(cfg: ExperimentConfig, fn_lab=None, verbose=True) -> SimulatedData:
    """Run the full simulation: positions with errors, rocking stack, direct 3D FFT, Poisson noise."""
    geom = cfg.derived()
    fn_lab = fn_lab or make_particle()
    nx = default_nx(cfg, geom)
    thetas, step = rocking_angles(cfg, geom, nx)
    XX, YY, nominal = nominal_positions(cfg, geom)
    n_pos = XX.size
    rng = np.random.default_rng(cfg.seed)
    err = rng.normal(size=(n_pos, 3)) * np.array([cfg.pos_err_uv_vox, cfg.pos_err_uv_vox, cfg.pos_err_w_vox])
    true_vox = nominal + err

    P = make_probe_volume(cfg, geom)
    U, V, W = _sim_grid(cfg.n_det, cfg.patch_w, geom)
    grid_beam = jnp.stack([U, V, W], axis=-1)  # (N, N, Nw, 3) beam-frame coordinates (m)
    B = beam_basis_matrix(geom.theta_B)
    pitches = jnp.asarray([geom.du, geom.dv, geom.dw])
    Nw = cfg.patch_w

    @jax.jit
    def one_pattern(offset_vox, theta):
        r_obj = grid_beam + offset_vox * pitches  # object-frame (beam axes) coordinates
        r_lab = r_obj @ B  # = B^T r (row-vector convention)
        r_src = r_lab @ rotation_matrix_y(theta)  # R_y(theta)^T r: physical rotation of the sample
        A = fn_lab(r_src[..., 0], r_src[..., 1], r_src[..., 2])
        dQ = delta_Q_beam_frame(theta, geom.theta_B, geom.wavelength)
        ramp = jnp.exp(2j * jnp.pi * (r_obj @ dQ))  # rocking: carrier change dQ.r
        f = P * A * ramp
        F = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(f), norm="ortho"))
        return jnp.abs(F[:, :, Nw // 2]) ** 2 * Nw  # detector plane q_w = 0 (ortho -> 2D-ortho of the w-sum)

    clean = np.zeros((n_pos, cfg.n_rock, cfg.n_det, cfg.n_det), np.float64)
    for i in range(n_pos):
        for k, t in enumerate(thetas):
            clean[i, k] = np.asarray(one_pattern(jnp.asarray(true_vox[i]), float(t)))
        if verbose and (i % 5 == 0 or i == n_pos - 1):
            print(f"  simulated position {i+1}/{n_pos}")

    scale = 1.0
    if cfg.photons_max is not None:
        scale = cfg.photons_max / clean.sum(axis=(2, 3)).max()
        measured = np.random.default_rng(cfg.seed + 1).poisson(clean * scale).astype(np.float64)
    else:
        measured = clean.copy()

    # object (envelope, beam grid, at the origin) for comparison with the reconstruction
    obj_shape = (cfg.n_det + 2 * cfg.obj_margin,) * 2 + (cfg.patch_w,)
    from .reference import lab_function_on_beam_grid

    true_obj = np.asarray(
        lab_function_on_beam_grid(fn_lab, obj_shape, (geom.du, geom.dv, geom.dw), geom.theta_B)
    )
    probe_2d = gaussian_probe_2d(cfg.n_det, geom.du, cfg.beam_fwhm)
    return SimulatedData(cfg, geom, thetas, np.stack([XX, YY], 1), nominal, true_vox,
                         measured.astype(np.float32), clean.astype(np.float32), float(scale),
                         probe_2d, true_obj, obj_shape, nx, {"rock_step": step})
