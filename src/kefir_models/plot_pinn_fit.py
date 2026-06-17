"""Dedicated plotting script for the logistic PINN fits.

Produces two figures:

1. A step-by-step overlay plot (one PNG per step) showing:
   - Observed trial data
   - Deterministic PINN fitted curve
   - Stochastic PINN drift curve
   - Stochastic-PINN SDE mean + confidence band

2. A side-by-side training-loss figure for the deterministic and
   stochastic PINN models.

Usage (from project root)::

    kefir-plot-pinn-fit \\
        --comparison-csv  logistic_pinn_outputs/water_kefir_logistic_pinn_comparison.csv \\
        --dense-csv       logistic_pinn_outputs/water_kefir_logistic_pinn_dense.csv \\
        --det-loss-csv    logistic_pinn_outputs/deterministic_logistic_pinn_training_loss.csv \\
        --sto-loss-csv    logistic_pinn_outputs/stochastic_logistic_pinn_training_loss.csv \\
        --output-dir      logistic_pinn_outputs \\
        --interval-level  0.90
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from kefir_models.plot_style import (
    COMPARISON_FIGURE_SIZE,
    COMPARISON_LAYOUT,
    LOSS_COLORS,
    LOSS_FIGURE_SIZE,
    LOSS_LAYOUT,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_comparison_axis_style,
    deduplicated_legend,
    model_line_style,
    save_fixed_layout_png,
    save_tight_png,
    trial_line_style,
)


def _infer_trial_columns(frame: pd.DataFrame) -> list[str]:
    """Return trial base names from columns ending with '_observed'."""
    return [
        col.replace("_observed", "")
        for col in frame.columns
        if col.endswith("_observed") and col != "observed_mean"
    ]


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column with invalid values dropped."""

    return pd.to_numeric(frame[column], errors="coerce").dropna()


# ---------------------------------------------------------------------------
# Step-by-step comparison plot
# ---------------------------------------------------------------------------


def save_pinn_comparison_plot(
    comparison_frame: pd.DataFrame,
    dense_frame: pd.DataFrame | None,
    trial_columns: list[str],
    interval_level: float,
    output_dir: Path,
) -> None:
    """Save observed data + PINN fits as step-by-step PNG files."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axis = plt.subplots(figsize=COMPARISON_FIGURE_SIZE)
    fig.subplots_adjust(**COMPARISON_LAYOUT)

    obs_time = comparison_frame["time"]

    # Use the dense grid for smooth curves when available; fall back to
    # the observation time grid.
    if dense_frame is not None and not dense_frame.empty:
        smooth_time = dense_frame["time"]
        det_curve = dense_frame.get(
            "deterministic_pinn",
            comparison_frame.get("deterministic_pinn"),
        )
        sto_drift = dense_frame.get(
            "stochastic_pinn_drift",
            comparison_frame.get("stochastic_pinn_drift"),
        )
        sde_mean = dense_frame.get("sde_mean", comparison_frame.get("sde_mean"))
        sde_lower = dense_frame.get("sde_lower", comparison_frame.get("sde_lower"))
        sde_upper = dense_frame.get("sde_upper", comparison_frame.get("sde_upper"))
    else:
        smooth_time = obs_time
        det_curve = comparison_frame.get("deterministic_pinn")
        sto_drift = comparison_frame.get("stochastic_pinn_drift")
        sde_mean = comparison_frame.get("sde_mean")
        sde_lower = comparison_frame.get("sde_lower")
        sde_upper = comparison_frame.get("sde_upper")

    # Pre-compute axis limits for a stable layout across all steps
    all_y_cols = [
        "observed_mean",
        "deterministic_pinn",
        "stochastic_pinn_drift",
        "sde_mean",
        "sde_lower",
        "sde_upper",
    ]
    for col in trial_columns:
        all_y_cols.append(f"{col}_observed")
    y_series = [
        _numeric_series(comparison_frame, column)
        for column in all_y_cols
        if column in comparison_frame.columns
    ]
    if dense_frame is not None and not dense_frame.empty:
        y_series.extend(
            _numeric_series(dense_frame, column)
            for column in [
                "deterministic_pinn",
                "stochastic_pinn_drift",
                "sde_mean",
                "sde_lower",
                "sde_upper",
            ]
            if column in dense_frame.columns
        )
    y_min = min(float(series.min()) for series in y_series if not series.empty)
    y_max = max(float(series.max()) for series in y_series if not series.empty)
    y_range = y_max - y_min if y_max > y_min else 1.0
    axis.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

    x_min, x_max = obs_time.min(), obs_time.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    axis.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    apply_comparison_axis_style(axis)

    def _save_step(step_num: int, suffix: str, title: str) -> None:
        axis.set_title(title)
        deduplicated_legend(axis)
        step_path = output_dir / f"{step_num:02d}_pinn_fit_{suffix}.png"
        save_fixed_layout_png(fig, step_path)
        print(f"Saved step {step_num}: {step_path}")

    # ------------------------------------------------------------------
    # Step 1 – observed data
    # ------------------------------------------------------------------
    for idx, col in enumerate(trial_columns):
        col_obs = f"{col}_observed"
        if col_obs not in comparison_frame.columns:
            continue
        axis.plot(
            obs_time,
            comparison_frame[col_obs],
            **trial_line_style(idx),
            label=col.replace("_", " ").title(),
        )
    _save_step(1, "data", "Water Kefir: Observed Trials")

    # ------------------------------------------------------------------
    # Step 2 – deterministic PINN fit
    # ------------------------------------------------------------------
    if det_curve is not None:
        axis.plot(
            smooth_time,
            det_curve,
            **model_line_style("deterministic_pinn"),
            label=MODEL_LABELS["deterministic_pinn"],
        )
    _save_step(2, "det_pinn", "Water Kefir: Deterministic PINN Fit")

    # ------------------------------------------------------------------
    # Step 3 – stochastic PINN drift
    # ------------------------------------------------------------------
    if sto_drift is not None:
        axis.plot(
            smooth_time,
            sto_drift,
            **model_line_style("stochastic_pinn_drift"),
            label=MODEL_LABELS["stochastic_pinn_drift"],
        )
    _save_step(3, "sto_pinn_drift", "Water Kefir: Stochastic PINN Drift")

    # ------------------------------------------------------------------
    # Step 4 – SDE mean + confidence band
    # ------------------------------------------------------------------
    if sde_mean is not None:
        axis.plot(
            smooth_time,
            sde_mean,
            **model_line_style("logistic_pinn_sde"),
            label=MODEL_LABELS["logistic_pinn_sde"],
        )
    if sde_lower is not None and sde_upper is not None:
        axis.fill_between(
            smooth_time,
            sde_lower,
            sde_upper,
            color=MODEL_COLORS["logistic_pinn_sde_band"],
            alpha=0.14,
            label=f"Logistic\nSDE PINN\n{interval_level:.0%} C.B.",
        )
        axis.plot(
            smooth_time,
            sde_lower,
            color=MODEL_COLORS["logistic_pinn_sde_band"],
            linewidth=0.9,
            alpha=0.8,
        )
        axis.plot(
            smooth_time,
            sde_upper,
            color=MODEL_COLORS["logistic_pinn_sde_band"],
            linewidth=0.9,
            alpha=0.8,
        )
    _save_step(4, "sde_band", "Water Kefir: Logistic PINN Fits")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Training-loss figure
# ---------------------------------------------------------------------------


def save_loss_plot(
    det_loss: pd.DataFrame | None,
    sto_loss: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Save deterministic and stochastic PINN training-loss histories."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(1, 2, figsize=LOSS_FIGURE_SIZE)
    fig.subplots_adjust(**LOSS_LAYOUT)

    titles = ["Deterministic PINN", "Stochastic PINN"]
    frames = [det_loss, sto_loss]

    # Columns to plot per model (skip epoch and the learned parameters)
    _skip = {"epoch", "growth_rate", "carrying_capacity", "diffusion"}

    for ax, title, frame in zip(axes, titles, frames):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)

        if frame is None or frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        x_col = "epoch" if "epoch" in frame.columns else None
        x = (
            pd.to_numeric(frame[x_col], errors="coerce")
            if x_col
            else pd.Series(range(1, len(frame) + 1))
        )

        loss_cols = [
            c for c in frame.columns if c not in _skip and pd.api.types.is_numeric_dtype(frame[c])
        ]

        all_vals = []
        for col in loss_cols:
            y = pd.to_numeric(frame[col], errors="coerce")
            valid = x.notna() & y.notna() & (y > 0)
            if not valid.any():
                continue
            all_vals.append(y[valid])
            ax.plot(
                x[valid],
                y[valid],
                color=LOSS_COLORS.get(col),
                linewidth=1.6,
                label=col.replace("_", " ").title(),
            )

        if all_vals:
            ax.set_yscale("log")
            ax.legend(fontsize=8)

    fig.suptitle("Logistic PINN: Training Loss Histories", fontsize=13)
    save_tight_png(fig, output_path)
    plt.close(fig)
    print(f"Loss plot saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot logistic PINN fitting results (step-by-step comparison + loss histories)."
        ),
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/water_kefir_logistic_pinn_comparison.csv"),
        help="Path to the per-observation-time comparison CSV.",
    )
    parser.add_argument(
        "--dense-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/water_kefir_logistic_pinn_dense.csv"),
        help="Path to the dense-grid prediction CSV (for smooth curves).",
    )
    parser.add_argument(
        "--det-loss-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/deterministic_logistic_pinn_training_loss.csv"),
        help="Deterministic PINN training-loss CSV.",
    )
    parser.add_argument(
        "--sto-loss-csv",
        type=Path,
        default=Path("logistic_pinn_outputs/stochastic_logistic_pinn_training_loss.csv"),
        help="Stochastic PINN training-loss CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logistic_pinn_outputs"),
        help="Directory where output PNGs are written.",
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=0.90,
        help="Confidence-band level shown in the legend label (default: 0.90).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point called by kefir-plot-pinn-fit."""
    args = parse_args()

    if not args.comparison_csv.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {args.comparison_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_frame = pd.read_csv(args.comparison_csv)
    trial_columns = _infer_trial_columns(comparison_frame)
    if not trial_columns:
        raise ValueError("No trial columns found in the comparison CSV.")

    dense_frame = None
    if args.dense_csv.exists():
        dense_frame = pd.read_csv(args.dense_csv)

    det_loss = pd.read_csv(args.det_loss_csv) if args.det_loss_csv.exists() else None
    sto_loss = pd.read_csv(args.sto_loss_csv) if args.sto_loss_csv.exists() else None

    # --- comparison steps ---
    save_pinn_comparison_plot(
        comparison_frame,
        dense_frame,
        trial_columns,
        args.interval_level,
        args.output_dir,
    )

    # --- loss histories ---
    loss_path = args.output_dir / "pinn_training_losses.png"
    save_loss_plot(det_loss, sto_loss, loss_path)


if __name__ == "__main__":
    main()
