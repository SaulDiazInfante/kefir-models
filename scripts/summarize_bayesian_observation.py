"""Generate posterior checks, forecasting comparisons, and LaTeX tables."""

from __future__ import annotations

import json
import os
import shutil
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from investigate_bayesian_observation import predictive_summary

from kefir_models.bayesian_observation import ObservationProblem, log_median, reference_prior
from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.report_tables import latex_table as base_latex_table

latex_table = partial(base_latex_table, resize=False)

ROOT = Path("results/bayesian")
LABELS = {
    "reference": "Reference prior",
    "noise_half": "Half noise prior",
    "noise_double": "Double noise prior",
}


def load_results():
    rows = []
    checks = []
    for case in sorted(ROOT.glob("*/*/cutoff_*/metrics.csv")):
        metrics = pd.read_csv(case)
        if len(metrics) != 1 or metrics.prediction_seed.iloc[0] != 0:
            raise ValueError(f"Regenerate predictive mixture quantiles: {case}")
        metadata = json.loads((case.parent / "experiment.json").read_text())
        if not metadata["computation_passed"]:
            raise ValueError(f"MCMC criteria not met: {case.parent}")
        prior = case.parent.parent.parent.name
        rows.append({"prior": prior, **metrics.iloc[0].to_dict()})
        diag = pd.read_csv(case.parent / "posterior_summary.csv")
        checks.append(
            {
                "prior": prior,
                "kind": metadata["arguments"]["kind"],
                "cutoff": metadata["arguments"]["cutoff"],
                "max_rhat": diag.rank_split_rhat.max(),
                "min_bulk_ess": diag.bulk_ess.min(),
                "min_tail_ess": diag.tail_ess.min(),
                "warmup": metadata["arguments"]["warmup"],
                "draws_per_chain": metadata["arguments"]["draws"],
                "acceptance_min": min(metadata["acceptance_by_chain"]),
                "acceptance_max": max(metadata["acceptance_by_chain"]),
            }
        )
    if len(rows) != 16:
        raise ValueError(f"Expected 16 complete Bayesian cases, found {len(rows)}.")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "all_metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(ROOT / "computation_checks.csv", index=False)
    summaries = []
    for (prior, kind), group in metrics.loc[~metrics.full_fit].groupby(["prior", "kind"]):
        n = group.evaluation_count
        summaries.append(
            {
                "prior": prior,
                "kind": kind,
                "rmse": np.sqrt(np.average(group.rmse**2, weights=n)),
                **{
                    key: np.average(group[key], weights=n)
                    for key in [
                        "coverage",
                        "mean_width",
                        "marginal_predictive_nll",
                        "parameter_log_variance",
                        "latent_log_variance",
                        "measurement_log_variance",
                    ]
                },
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(ROOT / "holdout_summary.csv", index=False)
    return metrics, summary, pd.DataFrame(checks)


def plugin_comparison(frame, columns):
    rows = []
    for kind in ["iid", "ou"]:
        for cutoff in [108, 120, 132]:
            directory = ROOT / "reference" / kind / f"cutoff_{cutoff}"
            chains = np.load(directory / "posterior_chains.npz")["chains"]
            point = np.median(chains.reshape(-1, chains.shape[-1]), axis=0)
            training = frame.loc[frame.time <= cutoff]
            evaluation = frame.loc[frame.time > cutoff]
            problem = ObservationProblem(
                training.time.to_numpy() / 12, training[columns].to_numpy(), kind
            )
            score, _ = predictive_summary(
                problem,
                point[None, None, :],
                evaluation.time.to_numpy() / 12,
                evaluation[columns].to_numpy(),
                1,
            )
            rows.append(
                {
                    "kind": kind,
                    "cutoff": cutoff,
                    "evaluation_count": len(evaluation) * len(columns),
                    **score,
                }
            )
    plugin = pd.DataFrame(rows)
    plugin.to_csv(ROOT / "fixed_parameter_comparison.csv", index=False)
    return plugin


def numerical_prediction_checks(frame, columns):
    rows = []
    for cutoff in [108, 120, 132]:
        forecast = pd.read_csv(
            ROOT / "reference/ou" / f"cutoff_{cutoff}" / "posterior_predictive.csv"
        )
        observed = forecast[[f"{column}_observed" for column in columns]].to_numpy()
        mean = forecast[[f"{column}_mean" for column in columns]].to_numpy()
        lower = forecast[[f"{column}_lower" for column in columns]].to_numpy()
        upper = forecast[[f"{column}_upper" for column in columns]].to_numpy()
        rows.append(
            {
                "cutoff": cutoff,
                "evaluation_count": observed.size,
                "mean_observed_minus_predicted": float((observed - mean).mean()),
                "below_lower": int((observed < lower).sum()),
                "above_upper": int((observed > upper).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "forecast_bias.csv", index=False)
    training = frame.loc[frame.time <= 132]
    evaluation = frame.loc[frame.time > 132]
    problem = ObservationProblem(training.time.to_numpy() / 12, training[columns].to_numpy(), "ou")
    chains = np.load(ROOT / "reference/ou/cutoff_132/posterior_chains.npz")["chains"]
    rows = []
    baseline = None
    for count in [6000, 12000, 24000]:
        score, prediction = predictive_summary(
            problem, chains, evaluation.time.to_numpy() / 12, evaluation[columns].to_numpy(), count
        )
        if baseline is None:
            baseline = prediction
        rows.append(
            {
                "integration_draws": count,
                **score,
                "max_lower_shift_from_6000": float(
                    np.max(np.abs(prediction["lower"] - baseline["lower"]))
                ),
                "max_upper_shift_from_6000": float(
                    np.max(np.abs(prediction["upper"] - baseline["upper"]))
                ),
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "predictive_integration_refinement.csv", index=False)


def figures(frame, metrics):
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharey=True)
    for axis, cutoff in zip(axes, [108, 120, 132]):
        for kind, color in [("iid", "#ba7029"), ("ou", "#287b9c")]:
            pred = pd.read_csv(
                ROOT / "reference" / kind / f"cutoff_{cutoff}" / "posterior_predictive.csv"
            )
            axis.fill_between(
                pred.time, pred.trial_1_lower, pred.trial_1_upper, color=color, alpha=0.16
            )
            axis.plot(pred.time, pred.trial_1_mean, color=color, label=f"Bayesian {kind.upper()}")
        future = frame.loc[frame.time > cutoff]
        axis.scatter(
            future.time, future.trial_1, color="black", s=24, label="Held-out trial 1", zorder=5
        )
        axis.set_title(f"Training through {cutoff}")
        axis.set_xlabel("Time (original units)")
        axis.grid(alpha=0.2)
        axis.set_ylabel("Response (original units)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "forecast_comparison.png", dpi=180)
    plt.close(fig)

    chains = np.load(ROOT / "reference/ou/cutoff_168/posterior_chains.npz")["chains"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    names = [
        "log initial",
        "log capacity margin",
        "log rate per interval",
        "log OU decay",
        "log process SD",
        "log measurement SD",
    ]
    for index, axis in enumerate(axes.flat):
        for chain in chains:
            axis.plot(np.arange(0, len(chain), 20), chain[::20, index], linewidth=0.4, alpha=0.7)
        axis.set_title(names[index], fontsize=10)
        axis.set_xlabel("Retained iteration")
    fig.tight_layout()
    fig.savefig(ROOT / "posterior_trace.png", dpi=150)
    plt.close(fig)

    flat = chains.reshape(-1, 6)[::12]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].scatter(np.exp(flat[:, 4]), np.exp(flat[:, 5]), s=3, alpha=0.15)
    axes[0].set_xlabel("Latent SD (log response)", fontsize=9)
    axes[0].set_ylabel("Measurement SD (log response)", fontsize=9)
    axes[1].scatter(12 / np.exp(flat[:, 3]), np.exp(flat[:, 4]), s=3, alpha=0.15)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Correlation time (original units)", fontsize=9)
    axes[1].set_ylabel("Latent SD (log response)", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "noise_identifiability.png", dpi=180)
    plt.close(fig)
    for name in ["forecast_comparison.png", "noise_identifiability.png"]:
        shutil.copy2(ROOT / name, Path("docs/assets/img") / f"bayesian_{name}")
    correlation = np.corrcoef(np.exp(flat[:, 4]), np.exp(flat[:, 5]))[0, 1]
    (ROOT / "noise_correlation.json").write_text(
        json.dumps({"process_measurement_sd_correlation": float(correlation)}, indent=2) + "\n"
    )

    # A proper, explicitly exploratory prior predictive check in original
    # numerical units; this is not evidence of prospective prior elicitation.
    rng = np.random.default_rng(33)
    prior = reference_prior("ou")
    z = rng.normal(prior.mean, prior.sd, size=(5000, 6))
    prior_draws = np.empty((5000, len(frame)))
    for i, point in enumerate(z):
        mean = log_median(frame.time.to_numpy() / 12, point, "ou")
        variance = np.exp(2 * point[4]) + np.exp(2 * point[5])
        prior_draws[i] = np.exp(mean + np.sqrt(variance) * rng.normal(size=len(frame)))
    lo, median, hi = np.quantile(prior_draws, [0.05, 0.5, 0.95], axis=0)
    pd.DataFrame({"time": frame.time, "q05": lo, "median": median, "q95": hi}).to_csv(
        ROOT / "prior_predictive.csv", index=False
    )


def write_tables(metrics, summary, checks, plugin):
    content = "% Generated by scripts/summarize_bayesian_observation.py; do not edit manually.\n"
    ref = metrics.loc[(metrics.prior == "reference") & (~metrics.full_fit)]
    content += latex_table(
        "Bayesian time holdouts. Intervals integrate parameter, latent-state, "
        "and measurement uncertainty.",
        "tab:bayesian-forecast",
        "llrrrr",
        "Model & Cutoff & RMSE & Coverage & Width & Marginal NLL",
        [
            f"{r.kind.upper()} & {r.cutoff:.0f} & {r.rmse:.4f} & {r.coverage:.4f} & "
            f"{r.mean_width:.4f} & {r.marginal_predictive_nll:.4f}"
            for r in ref.sort_values(["kind", "cutoff"]).itertuples()
        ],
    )
    content += latex_table(
        "Noise-prior sensitivity for the latent OU model. "
        "Cutoff-weighted descriptive summaries reuse observations.",
        "tab:bayesian-prior",
        "lrrrr",
        "Prior & RMSE & Coverage & Width & Marginal NLL",
        [
            f"{LABELS[r.prior]} & {r.rmse:.4f} & {r.coverage:.4f} & "
            f"{r.mean_width:.4f} & {r.marginal_predictive_nll:.4f}"
            for r in summary.loc[summary.kind == "ou"].itertuples()
        ],
    )
    physical = pd.read_csv(ROOT / "reference/ou/cutoff_168/posterior_summary.csv").dropna(
        subset=["median"]
    )
    labels = {
        "initial": "Initial median $x_0$",
        "capacity": "Capacity $K$",
        "rate_original": "Rate $r$",
        "process_sd": "Latent SD $a$",
        "observation_sd": "Measurement SD $b$",
        "correlation_time_original": "Correlation time $12/\\lambda$",
    }
    content += latex_table(
        "Full-data OU parameter posterior: marginal 90 percent credible intervals "
        "in the stated numerical units.",
        "tab:bayesian-parameters",
        "lrrr",
        "Parameter & Median & 5th percentile & 95th percentile",
        [
            f"{labels[r.parameter]} & {r.median:.5f} & {r.q05:.5f} & {r.q95:.5f}"
            for r in physical.itertuples()
        ],
    )
    Path("docs/generated/bayesian_tables.tex").write_text(content, encoding="utf-8")
    ou = summary.loc[(summary.prior == "reference") & (summary.kind == "ou")].iloc[0]
    iid = summary.loc[(summary.prior == "reference") & (summary.kind == "iid")].iloc[0]
    group = plugin.loc[plugin.kind == "ou"]
    fixed_coverage = np.average(group.coverage, weights=group.evaluation_count)
    sensitivity = summary.loc[summary.kind == "ou"].coverage
    interpretation = f"""    The cutoff-weighted reference OU coverage is {ou.coverage:.4f}, with
    RMSE {ou.rmse:.4f} and mean predictive width {ou.mean_width:.4f}. The
    observation-only Bayesian comparator has coverage {iid.coverage:.4f}.
    For the same OU model with coefficients fixed at their marginal posterior
    medians, coverage is {fixed_coverage:.4f}; this isolates a plug-in comparison
    within that model, rather than attributing every change from the earlier
    neural fits to parameter uncertainty alone. Noise-prior sensitivity gives
    descriptive coverage ranging from {sensitivity.min():.4f} to {sensitivity.max():.4f}.

    All reported runs meet the declared computational thresholds: the largest
    monitored $\\widehat R$ is {checks.max_rhat.max():.4f}, minimum bulk ESS is
    {checks.min_bulk_ess.min():.0f}, and minimum tail ESS is {checks.min_tail_ess.min():.0f}.
    These checks support finite-run numerical reliability, not model adequacy.
    The noise split and correlation time remain uncertain. The intervals in
    Table~\\ref{{tab:bayesian-parameters}} are parameter credible intervals;
    the forecast bands in Figure~\\ref{{fig:bayesian-forecast}} are observation
    predictive intervals. The original residual AIC/BIC scores are not used
    as Bayesian model evidence.

    All seven uncovered responses across the reference OU holdouts exceed
    their upper bounds. Mean observed-minus-predicted residuals increase from
    approximately 1.28 to 2.25 response units across the cutoffs. This pattern
    motivates investigation of structural underprediction, rather than a
    claim that parameter integration has resolved calibration. A predictive
    mixture integration check using 6000, 12000, and 24000 posterior draws at
    cutoff 132 leaves coverage unchanged; interval endpoints shift by at most
    approximately 0.012 response units relative to the 6000-draw calculation.

    The holdouts still reuse 15 distinct scalar observations across 36 forecast
    entries. A near-nominal aggregate rate would not establish calibration on
    independent future experiments. The Bayesian extension is retained as an
    exploratory observation-model comparison, and REV-07 remains open until
    predictive adequacy is established with stronger validation.

\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{{../results/bayesian/forecast_comparison.png}}
    \\caption{{Posterior predictive means and central 90 percent observation
    intervals for trial 1. Black markers are held-out responses. Metrics use
    all three trials; intervals integrate the parameter posterior.}}
    \\label{{fig:bayesian-forecast}}
\\end{{figure}}
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{../results/bayesian/noise_identifiability.png}}
    \\caption{{Joint posterior noise and correlation parameters for the full-data
    OU model. Correlation of noise components and a broad correlation-time
    range limit a mechanistic interpretation of their separate estimates.}}
    \\label{{fig:bayesian-identifiability}}
\\end{{figure}}
"""
    Path("docs/generated/bayesian_interpretation.tex").write_text(interpretation, encoding="utf-8")


def main():
    metrics, summary, checks = load_results()
    frame, columns = validate_frame(read_trial_csv(Path("data/raw/waterKefirTrialsReferece.csv")))
    plugin = plugin_comparison(frame, columns)
    numerical_prediction_checks(frame, columns)
    figures(frame, metrics)
    write_tables(metrics, summary, checks, plugin)
    print(summary.to_string(index=False))
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
