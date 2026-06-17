import argparse
import os
from pathlib import Path

import pandas as pd


def _plot_training_history(axis, history_frame: pd.DataFrame | None) -> None:
    """Plot one or more training objective histories on an axis."""

    axis.set_title("MSE error")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE error")
    axis.grid(alpha=0.25)

    if history_frame is None or history_frame.empty:
        axis.text(
            0.5,
            0.5,
            "No training history found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return

    x_column = "epoch" if "epoch" in history_frame.columns else None
    if x_column is None:
        x_values = pd.Series(
            range(1, len(history_frame) + 1),
            index=history_frame.index,
            name="epoch",
        )
    else:
        x_values = pd.to_numeric(history_frame[x_column], errors="coerce")

    objective_columns = [
        column
        for column in history_frame.columns
        if column != x_column and pd.api.types.is_numeric_dtype(history_frame[column])
    ]

    if not objective_columns:
        axis.text(
            0.5,
            0.5,
            "No numeric objective columns found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return

    objective_values = []
    for column in objective_columns:
        y_values = pd.to_numeric(history_frame[column], errors="coerce")
        valid_values = x_values.notna() & y_values.notna()
        if not valid_values.any():
            continue

        objective_values.append(y_values[valid_values])
        axis.plot(
            x_values[valid_values],
            y_values[valid_values],
            linewidth=1.8,
            label=column.replace("_", " ").title(),
        )

    if not objective_values:
        axis.text(
            0.5,
            0.5,
            "No valid objective values found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        return

    all_objective_values = pd.concat(objective_values)
    if (all_objective_values > 0).all():
        axis.set_yscale("log")

    if len(objective_columns) > 1:
        axis.legend(fontsize=8)


def save_plot(
    output_frame: pd.DataFrame,
    trial_columns: list[str],
    path: Path,
    history_frame: pd.DataFrame | None = None,
) -> None:
    """Save observed-vs-fitted trajectories and training objective history."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    # pylint: disable=import-outside-toplevel

    fig, (fit_axis, history_axis) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    time = output_frame["time"]

    # A vibrant and accessible color palette
    colors = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
    markers = ["o", "s", "^", "v", "D", "p", "*", "X"]

    _plot_training_history(history_axis, history_frame)

    for index, column in enumerate(trial_columns):
        marker = markers[index % len(markers)]
        color = colors[index % len(colors)]
        fit_axis.plot(
            time,
            output_frame[f"{column}_observed"],
            marker=marker,
            linestyle="",
            markersize=10,
            color=color,
            markeredgecolor="black",
            alpha=0.65,
            label=f"{column} observed",
        )

    if "fitted" in output_frame.columns:
        fitted_values = output_frame["fitted"]
    elif "fitted_mean" in output_frame.columns:
        fitted_values = output_frame["fitted_mean"]
    else:
        fitted_columns = [f"{column}_fitted" for column in trial_columns]
        fitted_values = output_frame[fitted_columns].mean(axis=1)

    fit_axis.plot(
        time,
        fitted_values,
        color="#2364aa",
        linewidth=2.6,
        label="Neural ODE Fit",
    )

    fit_axis.set_title("Observed Data and Neural ODE Fit")
    fit_axis.set_xlabel("Time (hrs)")
    fit_axis.set_ylabel(r"Kefirt wet biomass $(\mathrm{g/L})$")
    fit_axis.grid(alpha=0.25)
    fit_axis.legend(fontsize=8, ncol=2)

    fig.suptitle("Water Kefir Trials: Neural ODE Fit", fontsize=14)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot observed vs fitted trajectories from ODE fit output.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/neural_ode_outputs/water_kefir_neural_ode_fit.csv"),
        help="Path to the fitted values CSV file.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("outputs/neural_ode_outputs/water_kefir_neural_ode_fit.png"),
        help="Path where the generated plot will be saved.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("outputs/neural_ode_outputs/training_loss.csv"),
        help="Path to the training objective history CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run plotting from the command line."""
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    output_frame = pd.read_csv(args.input_csv)

    # Infer trial columns
    # We look for columns ending with "_observed" and ignore "observed_mean"
    trial_columns = [
        col.replace("_observed", "")
        for col in output_frame.columns
        if col.endswith("_observed") and col != "observed_mean"
    ]

    if not trial_columns:
        raise ValueError("Could not find any trial columns in the input CSV.")

    history_frame = None
    if args.history_csv.exists():
        history_frame = pd.read_csv(args.history_csv)

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    save_plot(output_frame, trial_columns, args.output_plot, history_frame)
    print(f"Plot saved to {args.output_plot}")


if __name__ == "__main__":
    main()
