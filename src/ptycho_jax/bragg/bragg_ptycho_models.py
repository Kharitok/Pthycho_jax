"""
Describes forward model for bragg ptychography. The model is composed from
Sample_selector constructiong and scanning the sample
Probe_selector constructing a probe
Proparagator performing the propagation to the detector plane
Detector converting wavefields into the measured intensities and taking into account possible nonlinear responce and detector atefacts

The model first implemented as a point model - description for the single intensity and then transformed into batch model using jax vmap
"""

# %%
from typing import Callable, Dict, Tuple, TypedDict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, Int32

### for typechecking


# %%
def angular_spectrum_propagate(field0, wavelength, dx, dy, z):
    Ny, Nx = field0.shape
    fx = jnp.fft.fftfreq(Nx, d=dx)
    fy = jnp.fft.fftfreq(Ny, d=dy)
    FX, FY = jnp.meshgrid(fx, fy, indexing="xy")
    inv_lambda2 = (1.0 / wavelength) ** 2
    kz = jnp.sqrt(jnp.clip(inv_lambda2 - FX**2 - FY**2, 0.0))
    H = jnp.exp(1j * 2 * jnp.pi * kz * z)  # evanescent terms -> 0 automatically at clip
    return jnp.fft.ifft2(jnp.fft.fft2(field0) * H)


# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

filepath = "/gpfs/exfel/exp/MID/202425/p008476/scratch/Kostya/Results/run_474/2025_10_02_17_20_51/2025_10_02_22_34_15_run_474_results.h5"

with h5py.File(filepath, "r") as f:
    modes = f["modes"][:]
    dx = f["dx_m"]
    lm = f["wavelength_m"]
# %%


plt.imshow()
