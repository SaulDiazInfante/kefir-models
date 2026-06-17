"""Unit tests for :mod:`kefir_models.plot_style`."""

import matplotlib.pyplot as plt

from kefir_models import plot_style


def test_model_line_style_returns_copy_with_overrides():
    base_linewidth = plot_style.MODEL_LINE_STYLES["neural_ode"]["linewidth"]
    style = plot_style.model_line_style("neural_ode", linewidth=9.9)

    assert style["color"] == plot_style.MODEL_COLORS["neural_ode"]
    assert style["linewidth"] == 9.9
    # The shared style dictionary must not be mutated by overrides.
    assert plot_style.MODEL_LINE_STYLES["neural_ode"]["linewidth"] == base_linewidth


def test_trial_line_style_cycles_markers_and_has_no_line():
    first = plot_style.trial_line_style(0)
    wrapped = plot_style.trial_line_style(len(plot_style.TRIAL_MARKERS))
    assert first["marker"] == wrapped["marker"]
    assert first["color"] == wrapped["color"]
    assert first["linestyle"] == ""


def test_model_labels_contain_core_models():
    for key in ("observed_mean", "classical_logistic", "neural_ode", "neural_sde"):
        assert key in plot_style.MODEL_LABELS


def test_save_tight_png_writes_file(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])
    path = tmp_path / "tight.png"
    plot_style.save_tight_png(figure, path)
    plt.close(figure)
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_fixed_layout_png_writes_file(tmp_path):
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])
    path = tmp_path / "fixed.png"
    plot_style.save_fixed_layout_png(figure, path)
    plt.close(figure)
    assert path.exists()
    assert path.stat().st_size > 0
