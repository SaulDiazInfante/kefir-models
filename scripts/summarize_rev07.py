"""Aggregate REV-07 experiments without treating overlapping forecasts as independent."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [
    "wide_euler",
    "wide_moments",
    "compact_euler",
    "compact_moments",
    "compact_moments_regularized",
    "classical_observation_noise",
]
LABELS = [
    "Wide\nEuler",
    "Wide\n16 steps",
    "Small\nEuler",
    "Small\n16 steps",
    "Small\n16 steps\n+ L2",
    "Logistic\n+ observation\nnoise",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/rev07"))
    args = parser.parse_args()
    root = args.results_dir
    frames = [pd.read_csv(p) for p in sorted(root.glob("**/cutoff_*/metrics.csv"))]
    metrics = pd.concat(frames, ignore_index=True)
    counts = metrics.groupby(["model", "cutoff", "training_seed"]).size()
    if len(counts) != 48 or set(metrics.model) != set(MODELS):
        raise ValueError("Expected six models, four cutoffs, and two training seeds.")
    for (model, _, _), count in counts.items():
        if count != (1 if model == "classical_observation_noise" else 3):
            raise ValueError("Incomplete simulation-seed results.")
    metrics.to_csv(root / "all_metrics.csv", index=False)
    names = [
        "forecast_rmse",
        "forecast_coverage",
        "forecast_mean_width",
        "forecast_gaussian_proxy_nll",
        "conditional_gaussian_proxy_nll",
        "conditional_gaussian_coverage",
        "evaluation_count",
    ]
    # Average simulation randomness first; repeated seeds are not new data.
    by_fit = metrics.groupby(["model", "cutoff", "training_seed"], as_index=False)[names].mean()
    by_fit.to_csv(root / "by_fit.csv", index=False)
    rows = []
    for (model, seed), group in by_fit.loc[by_fit.cutoff < 168].groupby(["model", "training_seed"]):
        n = group.evaluation_count.to_numpy()
        rows.append(
            {
                "model": model,
                "training_seed": seed,
                "rmse": np.sqrt(np.average(group.forecast_rmse**2, weights=n)),
                "coverage": np.average(group.forecast_coverage, weights=n),
                "mean_width": np.average(group.forecast_mean_width, weights=n),
                "gaussian_proxy_nll": np.average(group.forecast_gaussian_proxy_nll, weights=n),
                "overlapping_forecast_entries": int(n.sum()),
                "distinct_heldout_scalar_observations": 15,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "holdout_summary.csv", index=False)
    print(
        summary.groupby("model")[["rmse", "coverage", "mean_width", "gaussian_proxy_nll"]]
        .agg(["mean", "min", "max"])
        .to_string()
    )
    full = (
        by_fit.loc[by_fit.cutoff == 168]
        .groupby("model")
        .forecast_coverage.agg(["mean", "min", "max"])
    )
    held = summary.groupby("model").coverage.agg(["mean", "min", "max"])
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharey=True)
    for axis, values, title in zip(
        axes,
        [full, held],
        ["Fit on all times (in-sample)", "Time holdouts (cutoffs 108, 120, 132)"],
    ):
        values = values.reindex(MODELS)
        axis.bar(
            np.arange(6),
            values["mean"],
            color=["#cf6f24", "#3376a3", "#e7ae79", "#72aacc", "#3b8c66", "#777777"],
            width=0.7,
        )
        axis.errorbar(
            np.arange(6),
            values["mean"],
            yerr=np.stack([values["mean"] - values["min"], values["max"] - values["mean"]]),
            fmt="none",
            color="black",
            capsize=4,
        )
        axis.axhline(0.9, color="black", linestyle="--", linewidth=1, label="Nominal 90%")
        axis.set_xticks(np.arange(6), LABELS, fontsize=9)
        axis.set_ylim(0, 1.03)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Interval coverage")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[1].legend(loc="upper right")
    fig.text(
        0.5,
        0.015,
        "Means over simulation seeds 8-10 and training seeds 7,17.\n"
        "Whiskers span training-seed means; they are not confidence intervals.\n"
        "Conditioning rows excluded; overlapping holdouts reuse observations.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(root / "calibration_summary.png", dpi=180)
    plt.close(fig)
    if root.resolve() == Path("results/rev07").resolve():
        shutil.copy2(
            root / "calibration_summary.png", "docs/assets/img/rev07_calibration_summary.png"
        )


if __name__ == "__main__":
    main()
