"""Probe plots must show calibrated spatial axes and full field dimensions."""

import numpy as np

import pipeline_nanomax_upd as base
from ptycho_jax.visualisation import nanomax_diagnostics as diagnostics


def test_reconstruction_probe_plot_has_physical_axes_and_field_size(
    tmp_path, monkeypatch
):
    captured = []
    original = base._save_figure

    def save(fig, path):
        captured.append(fig)
        original(fig, path)

    monkeypatch.setattr(base, "_save_figure", save)
    base._save_probe(
        np.ones((2, 4, 6), dtype=complex),
        tmp_path / "probe.png",
        "Probe",
        pixel_size_m=(2e-8, 3e-8),
    )
    fig = captured[0]
    images = [ax for ax in fig.axes if ax.images]
    assert len(images) == 4
    for ax in images:
        assert ax.get_xlabel() == "x (µm)" and ax.get_ylabel() == "y (µm)"
        np.testing.assert_allclose(
            ax.images[0].get_extent(), [-0.09, 0.09, -0.04, 0.04], atol=1e-12
        )
    assert "0.18" in fig._suptitle.get_text() and "0.08" in fig._suptitle.get_text()
    assert (tmp_path / "probe.png").stat().st_size > 1000


def test_diagnostic_probe_plots_label_all_modes_and_field_dimensions(
    tmp_path, monkeypatch
):
    captured = []
    original = diagnostics._save

    def save(fig, path, filename_prefix=""):
        captured.append((fig, path.name))
        original(fig, path, filename_prefix)

    monkeypatch.setattr(diagnostics, "_save", save)
    rng = np.random.default_rng(3)
    modes = rng.normal(size=(2, 4, 6)) + 1j * rng.normal(size=(2, 4, 6))
    weights = np.repeat(np.eye(2)[None], 3, axis=0)
    decomposition = diagnostics.orthogonalize_probe_ensemble(modes, weights)
    diagnostics._plot_modes(modes, decomposition, tmp_path, (2e-8, 3e-8))
    for fig, name in captured:
        if name == "coherent_mode_spectrum.png":
            continue
        for ax in [ax for ax in fig.axes if ax.images]:
            assert ax.get_xlabel() == "x (µm)" and ax.get_ylabel() == "y (µm)"
            np.testing.assert_allclose(
                ax.images[0].get_extent(), [-0.09, 0.09, -0.04, 0.04], atol=1e-12
            )
        assert "0.18" in fig._suptitle.get_text() and "0.08" in fig._suptitle.get_text()
