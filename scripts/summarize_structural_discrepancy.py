"""Score the frozen OU comparator and summarize the structural investigation."""

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
from investigate_bayesian_observation import predictive_summary
from investigate_structural_discrepancy import interval_score

from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.structural_discrepancy import StructuralProblem

ROOT = Path("results/structural")
MODELS = ["independent_ou", "shared_ou", "independent_matern32", "shared_matern32"]
LABELS = ["Independent OU", "Shared OU", "Independent Matérn 3/2", "Shared Matérn 3/2"]


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_comparator(frame, columns):
    """Reuse validated upstream samples, verifying immutable inference sources."""
    for prior, multiplier in [("reference", 1), ("noise_half", 0.5), ("noise_double", 2)]:
        for cutoff in [108, 120, 132, 168]:
            upstream = Path("results/bayesian") / prior / "ou" / f"cutoff_{cutoff}"
            manifest = json.loads((upstream / "experiment.json").read_text())
            if not manifest["computation_passed"] or manifest["data_sha256"] != hash_file(
                "data/raw/waterKefirTrialsReferece.csv"
            ):
                raise ValueError(f"Upstream posterior validation failed: {upstream}")
            # The experiment's predictive helper has gained a dispatch hook;
            # the OU target, sampler, and prior sources are unchanged.
            for source in ["src/kefir_models/bayesian_observation.py", "src/kefir_models/mcmc.py"]:
                sampling_hashes = (
                    manifest.get("sampling_source_sha256") or manifest["source_sha256"]
                )
                if sampling_hashes[source] != hash_file(source):
                    raise ValueError(f"Frozen inference source changed: {source}")
            directory = ROOT / prior / "independent_ou" / f"cutoff_{cutoff}"
            directory.mkdir(parents=True, exist_ok=True)
            chains = np.load(upstream / "posterior_chains.npz")["chains"]
            training = frame.loc[frame.time <= cutoff]
            problem = StructuralProblem(
                training.time.to_numpy() / 12,
                training[columns].to_numpy(),
                "ou",
                "independent",
                multiplier,
            )
            full = cutoff == 168
            targets = {"rollout": frame.iloc[1:] if full else frame.loc[frame.time > cutoff]}
            if not full:
                targets["one_step"] = targets["rollout"].iloc[:1]
            rows = []
            for task, evaluation in targets.items():
                observed = evaluation[columns].to_numpy()
                score, prediction = predictive_summary(
                    problem, chains, evaluation.time.to_numpy() / 12, observed, 12000
                )
                output = pd.DataFrame({"time": evaluation.time.to_numpy()})
                for j, column in enumerate(columns):
                    output[column + "_observed"] = observed[:, j]
                    for key in ["mean", "lower", "upper"]:
                        output[column + "_" + key] = prediction[key][:, j]
                if task == "rollout":
                    old = pd.read_csv(upstream / "posterior_predictive.csv")
                    np.testing.assert_allclose(
                        output.to_numpy(), old[output.columns].to_numpy(), atol=1e-9, rtol=1e-10
                    )
                    old_score = pd.read_csv(upstream / "metrics.csv").iloc[0]
                    for key in [
                        "rmse",
                        "coverage",
                        "mean_width",
                        "marginal_predictive_nll",
                        "joint_predictive_nll_per_entry",
                    ]:
                        np.testing.assert_allclose(
                            score[key], old_score[key], atol=1e-9, rtol=1e-10
                        )
                output.to_csv(directory / f"{task}_predictive.csv", index=False)
                rows.append(
                    {
                        "kernel": "ou",
                        "sharing": "independent",
                        "cutoff": cutoff,
                        "noise_multiplier": multiplier,
                        "full_fit": full,
                        "task": task,
                        "computation_passed": True,
                        "evaluation_count": observed.size,
                        **score,
                        "interval_score": interval_score(
                            observed, prediction["lower"], prediction["upper"]
                        ).mean(),
                        "mean_bias": (observed - prediction["mean"]).mean(),
                        "above_upper": (observed > prediction["upper"]).sum(),
                        "below_lower": (observed < prediction["lower"]).sum(),
                    }
                )
            pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
            (directory / "posterior_summary.csv").write_bytes(
                (upstream / "posterior_summary.csv").read_bytes()
            )
            (directory / "experiment.json").write_text(
                json.dumps(
                    {
                        "upstream": str(upstream),
                        "upstream_manifest_sha256": hash_file(upstream / "experiment.json"),
                        "upstream_chains_sha256": hash_file(upstream / "posterior_chains.npz"),
                        "prediction_source_sha256": {
                            str(p): hash_file(p)
                            for p in [
                                Path(__file__),
                                Path("scripts/investigate_bayesian_observation.py"),
                                Path("src/kefir_models/structural_discrepancy.py"),
                                Path("src/kefir_models/posterior_prediction.py"),
                            ]
                        },
                        "computation_passed": True,
                        "arguments": {
                            "kernel": "ou",
                            "sharing": "independent",
                            "cutoff": cutoff,
                            "noise_multiplier": multiplier,
                            "draws": manifest["arguments"]["draws"],
                            "warmup": manifest["arguments"]["warmup"],
                        },
                        "acceptance_by_chain": manifest["acceptance_by_chain"],
                    },
                    indent=2,
                )
                + "\n"
            )


def load_results():
    rows, checks = [], []
    for path in sorted(ROOT.glob("*/*/cutoff_*/metrics.csv")):
        prior, model = path.parents[2].name, path.parents[1].name
        metadata = json.loads(path.with_name("experiment.json").read_text())
        if not metadata["computation_passed"]:
            raise ValueError(f"Unreliable computation: {path}")
        rows.append(pd.read_csv(path).assign(prior=prior, model=model))
        diag = pd.read_csv(path.with_name("posterior_summary.csv"))
        checks.append(
            {
                "prior": prior,
                "model": model,
                "cutoff": metadata["arguments"]["cutoff"],
                "draws_per_chain": metadata["arguments"]["draws"],
                "max_rhat": diag.rank_split_rhat.max(),
                "min_bulk_ess": diag.bulk_ess.min(),
                "min_tail_ess": diag.tail_ess.min(),
                "reused_upstream": "upstream" in metadata,
            }
        )
    if len(rows) != 56:
        raise ValueError(f"Expected 56 posterior cases, found {len(rows)}.")
    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(ROOT / "all_metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(ROOT / "computation_checks.csv", index=False)
    summaries = []
    for task, subset in [
        (
            "rollout",
            metrics.loc[
                (~metrics.full_fit)
                & (metrics.task == "rollout")
                & metrics.cutoff.isin([108, 120, 132])
            ],
        ),
        ("one_step", metrics.loc[(metrics.prior == "reference") & (metrics.task == "one_step")]),
    ]:
        for (prior, model), group in subset.groupby(["prior", "model"]):
            weights = group.evaluation_count
            summaries.append(
                {
                    "task": task,
                    "prior": prior,
                    "model": model,
                    "evaluation_count": weights.sum(),
                    "rmse": np.sqrt(np.average(group.rmse**2, weights=weights)),
                    **{
                        key: np.average(group[key], weights=weights)
                        for key in [
                            "coverage",
                            "mean_width",
                            "marginal_predictive_nll",
                            "joint_predictive_nll_per_entry",
                            "interval_score",
                            "mean_bias",
                        ]
                    },
                    "above_upper": group.above_upper.sum(),
                    "below_lower": group.below_lower.sum(),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(ROOT / "forecast_summary.csv", index=False)
    parameters = []
    for prior in ["reference", "noise_half", "noise_double"]:
        for model in MODELS:
            directory = ROOT / prior / model / "cutoff_168"
            data = pd.read_csv(directory / "posterior_summary.csv").dropna(subset=["median"])
            parameters.append(data.assign(prior=prior, model=model))
    pd.concat(parameters).to_csv(ROOT / "full_data_parameters.csv", index=False)
    return metrics, summary


def audit_one_step_targets():
    audit = []
    for model in MODELS:
        targets = []
        for cutoff in [108, 120, 132, 144, 156]:
            path = ROOT / "reference" / model / f"cutoff_{cutoff}" / "one_step_predictive.csv"
            data = pd.read_csv(path)
            if len(data) != 1 or data.time.iloc[0] != cutoff + 12:
                raise ValueError(f"Unexpected one-step target: {path}")
            targets.append(data.time.iloc[0])
            for column in [c for c in data if c.endswith("_observed")]:
                audit.append(
                    {
                        "model": model,
                        "origin": cutoff,
                        "target": data.time.iloc[0],
                        "trial": column.removesuffix("_observed"),
                        "observed": data[column].iloc[0],
                    }
                )
        if len(set(targets)) != 5:
            raise ValueError("Repeated one-step target.")
    pd.DataFrame(audit).to_csv(ROOT / "one_step_target_audit.csv", index=False)


def integration_checks(frame, columns):
    rows = []
    for model in MODELS[1:]:
        sharing, kernel = model.split("_", 1)
        directory = ROOT / "reference" / model / "cutoff_132"
        chains = np.load(directory / "posterior_chains.npz")["chains"]
        training, evaluation = frame.loc[frame.time <= 132], frame.loc[frame.time > 132]
        problem = StructuralProblem(
            training.time.to_numpy() / 12, training[columns].to_numpy(), kernel, sharing
        )
        baseline = None
        for count in [6000, 12000, 24000]:
            score, prediction = predictive_summary(
                problem,
                chains,
                evaluation.time.to_numpy() / 12,
                evaluation[columns].to_numpy(),
                count,
            )
            if baseline is None:
                baseline = prediction
            rows.append(
                {
                    "model": model,
                    "draws": count,
                    **score,
                    "max_endpoint_shift_from_6000": max(
                        np.max(np.abs(prediction[key] - baseline[key]))
                        for key in ["lower", "upper"]
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "predictive_integration_refinement.csv", index=False)


def figures(frame, columns):
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharey=True)
    colors = ["#227b9e", "#c45a28", "#36954a", "#8b62a8"]
    for ax, cutoff in zip(axes, [108, 120, 132]):
        for model, label, color in zip(MODELS, LABELS, colors):
            pred = pd.read_csv(
                ROOT / "reference" / model / f"cutoff_{cutoff}" / "rollout_predictive.csv"
            )
            ax.fill_between(
                pred.time, pred.trial_1_lower, pred.trial_1_upper, color=color, alpha=0.10
            )
            ax.plot(pred.time, pred.trial_1_mean, color=color, label=label)
        future = frame.loc[frame.time > cutoff]
        ax.scatter(future.time, future.trial_1, c="black", s=20, zorder=5)
        ax.set(
            title=f"Training through {cutoff}",
            xlabel="Time (original units)",
            ylabel="Response (original units)",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "forecast_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    for ax, model, label in zip(
        axes, ["independent_ou", "shared_ou"], ["Independent OU", "Shared OU"]
    ):
        path = ROOT / "reference" / model / "cutoff_168" / "posterior_chains.npz"
        if not path.exists():
            path = Path("results/bayesian/reference/ou/cutoff_168/posterior_chains.npz")
        chain = np.load(path)["chains"].reshape(-1, 6)[::24]
        capacity = np.exp(chain[:, 0]) * (1 + np.exp(chain[:, 1]))
        ax.scatter(capacity, np.exp(chain[:, 4]), s=3, alpha=0.15)
        ax.set(title=label, ylabel="Discrepancy SD (log response)")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Population logistic capacity K (original units)")
    fig.tight_layout()
    fig.savefig(ROOT / "capacity_discrepancy.png", dpi=180)
    plt.close(fig)


def main():
    frame, columns = validate_frame(read_trial_csv(Path("data/raw/waterKefirTrialsReferece.csv")))
    frozen_comparator(frame, columns)
    _, summary = load_results()
    audit_one_step_targets()
    integration_checks(frame, columns)
    figures(frame, columns)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
