"""Calibrated probe-image extents and field-of-view labels."""

import numpy as np


def probe_display_geometry(shape, pixel_size_m=None):
    """Return centered image edges, axis labels and the full array field size.

    The field of view is the image size, not a beam waist or FWHM. Rectangular
    arrays and anisotropic pitches are supported. Without calibration use pixels.
    """
    ny, nx = shape
    if pixel_size_m is None:
        dy, dx, unit = 1.0, 1.0, "pixels"
    else:
        dy, dx = np.broadcast_to(np.asarray(pixel_size_m, dtype=float), (2,)) * 1e6
        if not np.isfinite([dy, dx]).all() or min(dy, dx) <= 0:
            raise ValueError("Probe pixel pitches must be finite and positive.")
        unit = "µm"
    extent = (-nx * dx / 2, nx * dx / 2, -ny * dy / 2, ny * dy / 2)
    description = f"Field of view: x={nx * dx:.4g} {unit}, y={ny * dy:.4g} {unit} ({nx}×{ny} pixels)"
    return extent, f"x ({unit})", f"y ({unit})", description
