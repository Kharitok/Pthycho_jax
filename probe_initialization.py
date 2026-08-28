# functions to init the probe


import numpy as np

####


def init_probe_simple_fft(
    I_det: np.ndarray, p_det: float, z_sd: float, z_fs: float, wavelength: float
) -> tuple[np.ndarray, float]:
    """
    Initializes the sample probe by pre-multiplying a defocus phase at the detector,
    followed by a single IFFT back to the sample plane.

    Parameters:
    -----------
    I_det      : 2D array -> Measured empty beam intensity at detector
    p_det      : float    -> Detector pixel size (meters)
    z_sd       : float    -> Sample-to-detector distance (meters)
    z_fs       : float    -> Defocus: Focus-to-sample distance (+ for downstream, - for upstream)
    wavelength : float    -> X-ray wavelength (meters)

    Returns:
    --------
    P_sample   : 2D array -> Initialized probe at the sample plane
    p_sample   : float    -> Pixel size at the sample plane (meters)
    """
    Ny, Nx = I_det.shape

    # 1. Create physical coordinate grid at the detector
    x_d = (np.arange(Nx) - Nx / 2.0) * p_det
    y_d = (np.arange(Ny) - Ny / 2.0) * p_det
    X_d, Y_d = np.meshgrid(x_d, y_d)

    # 2. Calculate the defocus phase factor directly on the detector grid
    #    (This represents the free-space transfer function)
    phase = -(np.pi * z_fs) / (wavelength * z_sd**2) * (X_d**2 + Y_d**2)

    # 3. Apply phase factor to the detector amplitude (sqrt of intensity)
    E_det_mod = np.sqrt(np.maximum(I_det, 0)) * np.exp(1j * phase)

    # 4. Single IFFT back to the sample plane
    P_sample = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_det_mod)))

    # 5. Calculate resulting pixel size at the sample plane
    p_sample = (wavelength * z_sd) / (Nx * p_det)

    return P_sample, p_sample
