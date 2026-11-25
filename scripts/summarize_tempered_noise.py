"""Summarize the moment-controlled prior experiment without changing past fits."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from investigate_tempered_noise import forecast

from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.tempered_noise import TemperedNoiseProblem, median_matched_prior

ROOT = Path("results/tempered")
MODELS = ["independent_ou", "shared_ou", "independent_matern32", "shared_matern32"]
LABELS = ["Independent OU", "Shared OU", "Independent Matérn", "Shared Matérn"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_results():
    rows, checks, parameters = [], [], []
    expected = {
        (eta, model, cutoff)
        for eta in [2, 8]
        for model in MODELS
        for cutoff in ([108, 120, 132, 144, 156, 168] if eta == 2 else [108, 120, 132, 168])
    }
    found = set()
    for path in sorted(ROOT.glob("eta_*/*/cutoff_*/experiment.json")):
        meta = json.loads(path.read_text())
        args = meta["arguments"]
        eta = int(args["eta"])
        model = args["sharing"] + "_" + args["kernel"]
        cutoff = int(args["cutoff"])
        found.add((eta, model, cutoff))
        if not meta["computation_passed"] or meta["data_sha256"] != sha(
            "data/raw/waterKefirTrialsReferece.csv"
        ):
            raise ValueError(f"Invalid posterior case: {path}")
        for source, digest in meta["source_sha256"].items():
            if sha(source) != digest:
                raise ValueError(f"Source changed: {source}")
        diag = pd.read_csv(path.with_name("posterior_summary.csv"))
        functions = diag.loc[diag.parameter.str.startswith("predictive_mean_")]
        checks.append(
            {
                "eta": eta,
                "model": model,
                "cutoff": cutoff,
                "draws_per_chain": args["draws"],
                "max_rhat": diag.rank_split_rhat.max(),
                "min_bulk_ess": diag.bulk_ess.min(),
                "min_tail_ess": diag.tail_ess.min(),
                "min_predictive_mean_raw_ess": functions.raw_mean_ess.min(),
                "max_estimated_mean_mcse": functions.mcse_response_mean.max(),
                "max_chain_mean_range": (functions.chain_mean_max - functions.chain_mean_min).max(),
            }
        )
        if cutoff == 168:
            parameters.append(
                diag.loc[
                    ~diag.parameter.str.startswith(("predictive_mean_", "coordinate_"))
                ].assign(eta=eta, model=model)
            )
        else:
            rows.append(pd.read_csv(path.with_name("metrics.csv")).assign(model=model))
    if found != expected:
        raise ValueError(
            f"Expected exactly 40 cases; missing={expected - found}, extra={found - expected}"
        )
    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(ROOT / "all_metrics.csv", index=False)
    checks = pd.DataFrame(checks)
    checks.to_csv(ROOT / "computation_checks.csv", index=False)
    pd.concat(parameters, ignore_index=True).to_csv(ROOT / "full_data_parameters.csv", index=False)
    rows = []
    for task in ["rollout", "one_step"]:
        subset = metrics.loc[metrics.task == task]
        if task == "rollout":
            subset = subset.loc[subset.cutoff.isin([108, 120, 132])]
        else:
            subset = subset.loc[subset.eta == 2]
        for (eta, model), group in subset.groupby(["eta", "model"]):
            n = group.evaluation_count
            rows.append(
                {
                    "eta": eta,
                    "model": model,
                    "task": task,
                    "evaluation_count": n.sum(),
                    **{
                        key: np.average(group[key], weights=n)
                        for key in [
                            "coverage",
                            "mean_width",
                            "interval_score",
                            "marginal_predictive_nll",
                            "joint_predictive_nll_per_entry",
                            "bias_mean",
                            "bias_median",
                        ]
                    },
                    "rmse_mean": np.sqrt(np.average(group.rmse_mean**2, weights=n)),
                    "rmse_median": np.sqrt(np.average(group.rmse_median**2, weights=n)),
                    "above_upper": group.above_upper.sum(),
                    "below_lower": group.below_lower.sum(),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "forecast_summary.csv", index=False)
    old_path = Path("results/structural/forecast_summary.csv")
    old_manifest = json.loads(Path("results/structural/report_manifest.json").read_text())
    if sha(old_path) != old_manifest["source_and_artifact_sha256"][str(old_path)]:
        raise ValueError("Historical summary changed since its verified report.")
    old = pd.read_csv(old_path)
    old.loc[old.prior == "reference"].to_csv(ROOT / "historical_reference_scores.csv", index=False)
    return metrics, summary, checks


def prior_checks():
    rows = []
    for eta in [0.0, 2.0, 8.0]:
        prior = median_matched_prior(0.03, eta)
        rows.append(
            {
                "eta": eta,
                "kernel_location": prior.location,
                "kernel_scale": prior.scale,
                "log_normalizer_per_sd": prior.log_normalizer,
                "sd_q05": np.exp(prior.quantile(0.05)),
                "sd_median": np.exp(prior.quantile(0.5)),
                "sd_q95": np.exp(prior.quantile(0.95)),
                "cdf_at_target_median": prior.cdf(np.log(0.03)),
                "finite_positive_response_order_up_to_inclusive": np.sqrt(2 * eta),
                "mean_finite": eta >= 0.5,
                "second_moment_finite": eta >= 2.0,
                "fourth_moment_finite": eta >= 8.0,
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "prior_checks.csv", index=False)


def integration_checks():
    frame, columns = validate_frame(read_trial_csv(Path("data/raw/waterKefirTrialsReferece.csv")))
    training, evaluation = frame.loc[frame.time <= 132], frame.loc[frame.time > 132]
    rows = []
    for sharing in ["independent", "shared"]:
        model = sharing + "_ou"
        chains = np.load(ROOT / "eta_2" / model / "cutoff_132/posterior_chains.npz")["chains"]
        problem = TemperedNoiseProblem(
            training.time.to_numpy() / 12, training[columns].to_numpy(), "ou", sharing, 2.0
        )
        reference = None
        for count in [6000, 12000, 24000]:
            scores, prediction = forecast(
                problem,
                chains,
                evaluation.time.to_numpy() / 12,
                evaluation[columns].to_numpy(),
                count,
            )
            if reference is None:
                reference = prediction
            rows.append(
                {
                    "model": model,
                    "components": count,
                    **scores,
                    "max_mean_shift_from_6000": np.max(
                        np.abs(prediction.response_mean - reference.response_mean)
                    ),
                    "max_endpoint_shift_from_6000": max(
                        np.max(np.abs(prediction[key] - reference[key]))
                        for key in ["lower", "upper"]
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "predictive_integration_refinement.csv", index=False)


def figures():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    colors = ["#666666", "#237d9d", "#bb5e32"]
    for eta, color in zip([0.0, 2.0, 8.0], colors):
        prior = median_matched_prior(0.03, eta)
        for axis, limits in zip(axes, [(0.001, 0.3), (0.1, 3.0)]):
            sd = np.geomspace(*limits, 400)
            density = np.exp([prior.logpdf(z) - z for z in np.log(sd)])
            axis.plot(sd, density, color=color, label=f"eta={eta:g}")
            axis.set_xscale("log")
            axis.set_xlabel("Noise SD (log response)")
            axis.set_ylabel("Prior density for SD")
            axis.grid(alpha=0.2)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-18, 10)
    axes[0].set_title("Matched median: 0.03")
    axes[1].set_title("Changed upper tails")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "noise_prior_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharey=True)
    for ax, cutoff in zip(axes, [108, 120, 132]):
        for model, label, color in zip(
            MODELS, LABELS, ["#227b9e", "#c45a28", "#36954a", "#8b62a8"]
        ):
            pred = pd.read_csv(
                ROOT / "eta_2" / model / f"cutoff_{cutoff}" / "rollout_predictive.csv"
            )
            pred = pred.loc[pred.trial == "trial_3"]
            ax.plot(pred.time, pred.response_mean, color=color, label=label)
            ax.fill_between(pred.time, pred.lower, pred.upper, color=color, alpha=0.10)
        ax.scatter(pred.time, pred.observed, color="black", s=22, zorder=5)
        ax.set(
            title=f"Training through {cutoff}; trial 3",
            xlabel="Time (original units)",
            ylabel="Response (original units)",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "forecast_comparison.png", dpi=180)
    plt.close(fig)


def main():
    _, summary, _ = load_results()
    prior_checks()
    integration_checks()
    figures()
    (ROOT / "summary_lineage.json").write_text(
        json.dumps(
            {
                "script_sha256": sha(__file__),
                "historical_summary_sha256": sha("results/structural/forecast_summary.csv"),
                "data_sha256": sha("data/raw/waterKefirTrialsReferece.csv"),
                "cases": 40,
                "new_independent_units": 0,
                "prior_selection": (
                    "Two fixed strengths; no optimization against forecast coverage."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
