"""Refit one structural model with median-matched, tempered noise priors."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from investigate_bayesian_observation import curvature_covariance, parameter_draws
from scipy.optimize import minimize

from kefir_models.mcmc import diagnostics, effective_sample_size, metropolis_chain, split_chains
from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.posterior_prediction import gaussian_mixture_quantile
from kefir_models.prospective_validation import gaussian_forecast, score_mixture
from kefir_models.tempered_noise import TemperedNoiseProblem


def forecast(problem, chains, times, observed=None, count=12000):
    means, covariances = gaussian_forecast(problem, chains, times, count)
    variance = np.diagonal(covariances, axis1=1, axis2=2)
    response_mean = np.exp(means + variance / 2).mean(axis=0)
    if observed is not None:
        pointwise, joint = score_mixture(means, covariances, observed.ravel())
    else:
        pointwise = pd.DataFrame(
            {
                name: [
                    np.exp(gaussian_mixture_quantile(means[:, j], variance[:, j], probability))
                    for j in range(means.shape[1])
                ]
                for name, probability in [("lower", 0.05), ("point", 0.5), ("upper", 0.95)]
            }
        )
        joint = None
    pointwise["response_mean"] = response_mean
    score = {}
    if observed is not None:
        y = observed.ravel()
        score = {
            key: float(pointwise[key].mean())
            for key in ["coverage", "mean_width", "interval_score", "marginal_predictive_nll"]
        }
        score.update(
            {
                "evaluation_count": len(y),
                "rmse_mean": float(np.sqrt(np.mean((y - response_mean) ** 2))),
                "rmse_median": float(np.sqrt(pointwise.squared_error.mean())),
                "bias_mean": float((y - response_mean).mean()),
                "bias_median": float(pointwise.mean_bias.mean()),
                "joint_predictive_nll_per_entry": joint,
                "above_upper": int((y > pointwise.upper).sum()),
                "below_lower": int((y < pointwise.lower).sum()),
            }
        )
    return score, pointwise


def mean_diagnostics(problem, chains, times):
    # Monitor the actual integrand for the newly finite predictive mean.
    # A regularly spaced subchain reduces the cost of these likelihood-free
    # conditional solves. Its diagnostics retain that stride explicitly.
    stride = 8
    subchains = chains[:, ::stride]
    values = np.empty((*subchains.shape[:2], len(times), problem.values.shape[1]))
    for c, chain in enumerate(subchains):
        for d, z in enumerate(chain):
            mean, latent, noise = problem.conditional_log_distribution(z, times)
            values[c, d] = np.exp(mean + 0.5 * (latent.diagonal() + noise)[:, None])
    rows = []
    for i, prediction_time in enumerate(times):
        for j in range(problem.values.shape[1]):
            v = values[:, :, i, j]
            if not np.isfinite(v).all():
                raise ValueError("Predictive-mean functional overflow; inspect computation.")
            rows.append(
                {
                    "parameter": f"predictive_mean_time_{prediction_time:g}_trial_{j + 1}",
                    "mean": v.mean(),
                    "median": np.median(v),
                    "q05": np.quantile(v, 0.05),
                    "q95": np.quantile(v, 0.95),
                    **diagnostics(v),
                    "draw_stride": stride,
                    "chain_mean_min": v.mean(axis=1).min(),
                    "chain_mean_max": v.mean(axis=1).max(),
                    "raw_mean_ess": effective_sample_size(split_chains(v)),
                    "mcse_response_mean": np.std(v, ddof=1)
                    / np.sqrt(effective_sample_size(split_chains(v))),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=["ou", "matern32"], default="ou")
    parser.add_argument("--sharing", choices=["shared", "independent"], default="independent")
    parser.add_argument("--eta", type=float, choices=[2.0, 8.0], default=2.0)
    parser.add_argument("--cutoff", type=float, default=168.0)
    parser.add_argument("--warmup", type=int, default=6000)
    parser.add_argument("--draws", type=int, default=36000)
    parser.add_argument("--prediction-draws", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-chains", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path("data/raw/waterKefirTrialsReferece.csv")
    frame, columns = validate_frame(read_trial_csv(data_path))
    training = frame.loc[frame.time <= args.cutoff]
    if len(training) < 2 or training.time.iloc[-1] != args.cutoff:
        raise ValueError("Cutoff must be an observed time.")
    problem = TemperedNoiseProblem(
        training.time.to_numpy() / 12,
        training[columns].to_numpy(),
        args.kernel,
        args.sharing,
        args.eta,
    )
    sources = [
        Path(__file__),
        Path("scripts/investigate_bayesian_observation.py"),
        *[
            Path("src/kefir_models") / name
            for name in [
                "tempered_noise.py",
                "structural_discrepancy.py",
                "bayesian_observation.py",
                "mcmc.py",
                "posterior_prediction.py",
                "prospective_validation.py",
            ]
        ],
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if args.reuse_chains:
        old = json.loads((args.output_dir / "experiment.json").read_text())
        if old["source_sha256"] != hashes or old["data_sha256"] != data_hash:
            raise ValueError("Sources or data changed; refit instead of reusing.")
        for key in ["kernel", "sharing", "eta", "cutoff", "warmup", "draws", "seed"]:
            if old["arguments"][key] != getattr(args, key):
                raise ValueError(f"Cannot reuse changed {key}.")
        chains = np.load(args.output_dir / "posterior_chains.npz")["chains"]
        acceptance = old["acceptance_by_chain"]
    else:
        rng = np.random.default_rng(args.seed)
        modes = [
            minimize(
                lambda z: -problem.log_posterior(z),
                initial,
                method="BFGS",
                options={"maxiter": 1000, "gtol": 1e-5},
            )
            for initial in [problem.prior.mean, problem.prior.mean + rng.normal(0, 0.4, 6)]
        ]
        point = min(modes, key=lambda m: m.fun).x
        covariance = curvature_covariance(problem.log_posterior, point)
        factor = np.linalg.cholesky(covariance)
        chains, acceptance = [], []
        for c in range(4):
            result = metropolis_chain(
                problem.log_posterior,
                point + 2 * factor @ rng.normal(size=6),
                covariance,
                args.warmup,
                args.draws,
                args.seed + 1009 * (c + 1),
            )
            chains.append(result.draws)
            acceptance.append(result.acceptance)
            print(
                f"{args.sharing} {args.kernel} eta={args.eta} cutoff={args.cutoff} "
                f"chain={c + 1} acceptance={result.acceptance:.3f}",
                flush=True,
            )
        chains = np.asarray(chains)
        np.savez_compressed(args.output_dir / "posterior_chains.npz", chains=chains)
    rows = []
    for name, values in parameter_draws(chains, "ou").items():
        rows.append(
            {
                "parameter": name,
                "mean": values.mean(),
                "median": np.median(values),
                "q05": np.quantile(values, 0.05),
                "q95": np.quantile(values, 0.95),
                **diagnostics(values),
                "draw_stride": 1,
            }
        )
    for i in range(6):
        rows.append(
            {"parameter": f"coordinate_{i}", **diagnostics(chains[:, :, i]), "draw_stride": 1}
        )
    full = args.cutoff == float(frame.time.max())
    future = (
        np.array([args.cutoff + 12])
        if full
        else frame.loc[frame.time > args.cutoff].time.to_numpy()
    )
    rows.extend(mean_diagnostics(problem, chains, np.unique(future[[0, -1]]) / 12))
    diag = pd.DataFrame(rows)
    diag.to_csv(args.output_dir / "posterior_summary.csv", index=False)
    passed = bool(
        (diag.rank_split_rhat < 1.01).all()
        and (diag.bulk_ess >= 400).all()
        and (diag.tail_ess >= 400).all()
        and (diag.raw_mean_ess.dropna() >= 400).all()
    )
    scores = []
    tasks = {"future": future} if full else {"rollout": future, "one_step": future[:1]}
    for task, times in tasks.items():
        observed = None if full else frame.loc[frame.time.isin(times), columns].to_numpy()
        score, prediction = forecast(problem, chains, times / 12, observed, args.prediction_draws)
        prediction.insert(0, "trial", np.tile(columns, len(times)))
        prediction.insert(0, "time", np.repeat(times, len(columns)))
        prediction.to_csv(args.output_dir / f"{task}_predictive.csv", index=False)
        if not full:
            scores.append(
                {
                    "kernel": args.kernel,
                    "sharing": args.sharing,
                    "eta": args.eta,
                    "cutoff": args.cutoff,
                    "task": task,
                    "computation_passed": passed,
                    **score,
                }
            )
    pd.DataFrame(scores).to_csv(args.output_dir / "metrics.csv", index=False)
    prior = problem.noise_prior
    metadata = {
        "arguments": vars(args),
        "data_sha256": data_hash,
        "source_sha256": hashes,
        "computation_passed": passed,
        "acceptance_by_chain": acceptance,
        "gaussian_kernel_location": problem.prior.mean.tolist(),
        "gaussian_kernel_scale": problem.prior.sd.tolist(),
        "noise_prior": {
            "family": "median_matched_tempered_lognormal",
            "eta": args.eta,
            "median": 0.03,
            "log_normalizer_per_sd": prior.log_normalizer,
            "gaussian_kernel_location_per_log_sd": prior.location,
            "log_scale": prior.scale,
        },
        "finite_response_moments_up_to_inclusive": float(np.sqrt(2 * args.eta)),
        "predictive_mean_diagnostic_stride": 8,
        "prediction_draws": args.prediction_draws,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "elapsed_seconds": time.monotonic() - start,
        "units": "Original numerical units; physical units unconfirmed.",
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(
        diag[["parameter", "rank_split_rhat", "bulk_ess", "tail_ess"]].to_string(index=False),
        flush=True,
    )
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
