"""Shared plotting styles for kefir model comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_COLORS = {
    "observed_mean": "#111111",
    "classical_logistic": "#2a9d8f",
    "neural_ode": "#2364aa",
    "neural_sde": "#e66101",
    "neural_sde_band": "#2dcdb0",
    "deterministic_pinn": "#6a3d9a",
    "stochastic_pinn_drift": "#c51b7d",
    "logistic_pinn_sde": "#8c2d04",
    "logistic_pinn_sde_band": "#fdae61",
}
MODEL_LINE_STYLES = {
    "observed_mean": {
        "color": MODEL_COLORS["observed_mean"],
        "linewidth": 2.0,
        "linestyle": ":",
    },
    "classical_logistic": {
        "color": MODEL_COLORS["classical_logistic"],
        "linewidth": 2.5,
        "linestyle": ":",
    },
    "neural_ode": {
        "color": MODEL_COLORS["neural_ode"],
        "linewidth": 2.5,
        "linestyle": "--",
    },
    "neural_sde": {
        "color": MODEL_COLORS["neural_sde"],
        "linewidth": 2.5,
        "linestyle": "-.",
    },
    "deterministic_pinn": {
        "color": MODEL_COLORS["deterministic_pinn"],
        "linewidth": 2.4,
        "linestyle": "-",
    },
    "stochastic_pinn_drift": {
        "color": MODEL_COLORS["stochastic_pinn_drift"],
        "linewidth": 2.4,
        "linestyle": "--",
    },
    "logistic_pinn_sde": {
        "color": MODEL_COLORS["logistic_pinn_sde"],
        "linewidth": 2.4,
        "linestyle": "-.",
    },
}
MODEL_LABELS = {
    "observed_mean": "Observed\nmean",
    "classical_logistic": "Logistic\nODE",
    "neural_ode": "NODE",
    "neural_sde": "NSDE\nmean",
    "deterministic_pinn": "Logistic\nODE PINN",
    "stochastic_pinn_drift": "Logistic\nPINN drift",
    "logistic_pinn_sde": "Logistic\nSDE PINN\nmean",
}
TRIAL_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
TRIAL_MARKERS = ["o", "s", "^", "v", "D", "p", "*", "X"]
LOSS_COLORS = {
    "total_loss": "#111111",
    "data_mse": MODEL_COLORS["neural_ode"],
    "physics_mse": MODEL_COLORS["classical_logistic"],
    "transition_nll": MODEL_COLORS["logistic_pinn_sde"],
}
REFERENCE_LINE_COLOR = "#777777"

COMPARISON_FIGURE_SIZE = (11.5, 5.9)
COMPARISON_LAYOUT = {
    "right": 0.68,
    "bottom": 0.15,
}
LEGEND_KWARGS = {
    "bbox_to_anchor": (1.04, 1.0),
    "loc": "upper left",
    "fontsize": 8.5,
}
OUTPUT_DPI = 300
FIXED_TIGHT_BBOX_INCHES = (0.80, 0.30, 9.45, 5.60)
LOSS_FIGURE_SIZE = (13.0, 5.0)
LOSS_LAYOUT = {
    "left": 0.08,
    "right": 0.98,
    "bottom": 0.14,
    "top": 0.84,
    "wspace": 0.25,
}
OBJECTIVE_FIGURE_SIZE = (14.5, 4.8)
OBJECTIVE_LAYOUT = {
    "left": 0.07,
    "right": 0.98,
    "bottom": 0.15,
    "top": 0.80,
    "wspace": 0.30,
}


def model_line_style(model_key: str, **overrides: Any) -> dict[str, Any]:
    """Return a copy of the shared line style for a model."""

    style = dict(MODEL_LINE_STYLES[model_key])
    style.update(overrides)
    return style


def trial_line_style(index: int, **overrides: Any) -> dict[str, Any]:
    """Return a consistent observed-trial marker style."""

    style = {
        "marker": TRIAL_MARKERS[index % len(TRIAL_MARKERS)],
        "linestyle": "",
        "markersize": 9.0,
        "color": TRIAL_COLORS[index % len(TRIAL_COLORS)],
        "markeredgecolor": "black",
        "alpha": 0.45,
    }
    style.update(overrides)
    return style


def deduplicated_legend(axis, **overrides: Any) -> None:
    """Place one legend entry per label outside the plotting area."""

    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_kwargs = dict(LEGEND_KWARGS)
    legend_kwargs.update(overrides)
    axis.legend(by_label.values(), by_label.keys(), **legend_kwargs)


def apply_comparison_axis_style(axis) -> None:
    """Apply common axis labels and grid styling for comparison figures."""

    axis.set_xlabel("Time (hrs)")
    axis.set_ylabel(r"Kefir wet biomass $(\mathrm{g/L})$")
    axis.grid(alpha=0.25)


def save_fixed_layout_png(fig, path: Path) -> None:
    """Save a tighter PNG using the same crop box for every comparison frame."""

    from matplotlib.transforms import Bbox  # pylint: disable=import-outside-toplevel

    fig.set_size_inches(*COMPARISON_FIGURE_SIZE, forward=True)
    fig.savefig(
        path,
        dpi=OUTPUT_DPI,
        bbox_inches=Bbox.from_extents(*FIXED_TIGHT_BBOX_INCHES),
    )


def save_tight_png(fig, path: Path) -> None:
    """Save a standalone figure with tight bounds."""

    fig.savefig(path, dpi=OUTPUT_DPI, bbox_inches="tight")
