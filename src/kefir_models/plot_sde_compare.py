import argparse
import os
from pathlib import Path

import pandas as pd

from kefir_models.plot_style import (
    COMPARISON_FIGURE_SIZE,
    COMPARISON_LAYOUT,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_comparison_axis_style,
    deduplicated_legend,
    model_line_style,
    save_fixed_layout_png,
    trial_line_style,
)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column with invalid values dropped."""

    return pd.to_numeric(frame[column], errors="coerce").dropna()


def save_comparison_plot(
    comparison_frame: pd.DataFrame,
    trial_columns: list[str],
    interval_level: float,
    path: Path,
) -> None:
    """Save observed data, ODE fit, and SDE predictive interval plot iteratively."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axis = plt.subplots(figsize=COMPARISON_FIGURE_SIZE)
    fig.subplots_adjust(**COMPARISON_LAYOUT)
    time = comparison_frame["time"]

    # Pre-calculate axis limits to fix the layout across the sequence
    all_y_cols = [
        "observed_mean",
        "classical_logistic_fitted_mean",
        "ode_fitted_mean",
        "sde_mean",
        "sde_lower",
        "sde_upper",
    ]
    for col in trial_columns:
        all_y_cols.append(f"{col}_observed")

    valid_cols = [c for c in all_y_cols if c in comparison_frame.columns]
    y_series = [_numeric_series(comparison_frame, column) for column in valid_cols]
    y_min = min(float(series.min()) for series in y_series if not series.empty)
    y_max = max(float(series.max()) for series in y_series if not series.empty)
    y_range = y_max - y_min if y_max > y_min else 1.0
    axis.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

    x_min, x_max = time.min(), time.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    axis.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    axis.set_title("Water Kefir: Neural ODE vs Neural SDE vs Logistic ODE")
    apply_comparison_axis_style(axis)

    def save_step(step_num: int, suffix: str):
        deduplicated_legend(axis)
        step_path = path.with_name(f"{step_num:02d}_{path.stem}_{suffix}{path.suffix}")
        save_fixed_layout_png(fig, step_path)
        print(f"Saved step {step_num}: {step_path}")

    # 1. Plot the data
    for index, column in enumerate(trial_columns):
        axis.plot(
            time,
            comparison_frame[f"{column}_observed"],
            **trial_line_style(index),
            label=f"{column}",
        )
    save_step(1, "data")

    # 2. Plot the mean of the data
    axis.plot(
        time,
        comparison_frame["observed_mean"],
        **model_line_style("observed_mean"),
        label=MODEL_LABELS["observed_mean"],
    )
    save_step(2, "obs_mean")

    # 3. Plot the classic fitting
    axis.plot(
        time,
        comparison_frame["classical_logistic_fitted_mean"],
        **model_line_style("classical_logistic"),
        label=MODEL_LABELS["classical_logistic"],
    )
    save_step(3, "classic_fit")

    # 4. Plot the NODE fitting
    axis.plot(
        time,
        comparison_frame["ode_fitted_mean"],
        **model_line_style("neural_ode"),
        label=MODEL_LABELS["neural_ode"],
    )
    save_step(4, "node_fit")

    # 5. Plot the estimated mean with SDE
    axis.plot(
        time,
        comparison_frame["sde_mean"],
        **model_line_style("neural_sde"),
        label=MODEL_LABELS["neural_sde"],
    )
    save_step(5, "sde_mean")

    # 6. Plot the confidence band
    if "sde_lower" in comparison_frame.columns and "sde_upper" in comparison_frame.columns:
        axis.fill_between(
            time,
            comparison_frame["sde_lower"],
            comparison_frame["sde_upper"],
            color=MODEL_COLORS["neural_sde_band"],
            alpha=0.15,
            label=f"NSDE\n{interval_level:.0%} C.B.",
        )
        axis.plot(
            time,
            comparison_frame["sde_lower"],
            color=MODEL_COLORS["neural_sde_band"],
            linewidth=0.9,
            alpha=0.75,
        )
        axis.plot(
            time,
            comparison_frame["sde_upper"],
            color=MODEL_COLORS["neural_sde_band"],
            linewidth=0.9,
            alpha=0.75,
        )

    save_step(6, "sde_band")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot comparison trajectories from SDE/ODE fit output.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("neural_sde_comparison_outputs/water_kefir_neural_dynamics_comparison.csv"),
        help="Path to the comparison values CSV file.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path(
            "neural_sde_comparison_outputs/water_kefir_neural_dynamics_comparison_replot.png",
        ),
        help="Path where the generated plot will be saved.",
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=0.90,
        help="Interval level used for the label (e.g. 0.90).",
    )
    return parser.parse_args()


def main() -> None:
    """Run plotting from the command line."""
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    comparison_frame = pd.read_csv(args.input_csv)

    # Infer trial columns
    # We look for columns ending with "_observed" and ignore "observed_mean"
    trial_columns = [
        col.replace("_observed", "")
        for col in comparison_frame.columns
        if col.endswith("_observed") and col != "observed_mean"
    ]

    if not trial_columns:
        raise ValueError("Could not find any trial columns in the input CSV.")

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    save_comparison_plot(comparison_frame, trial_columns, args.interval_level, args.output_plot)
    print(f"Plot saved to {args.output_plot}")


if __name__ == "__main__":
    main()
