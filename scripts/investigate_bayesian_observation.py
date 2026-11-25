"""Fit and validate Bayesian logistic observation models on fixed time holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp

from kefir_models.bayesian_observation import (
    ObservationProblem,
    log_median,
    physical_parameters,
)
from kefir_models.mcmc import diagnostics, metropolis_chain
from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.posterior_prediction import gaussian_mixture_quantile


def curvature_covariance(target, point):
    dimension = len(point)
    step = 1e-3
    hessian = np.empty((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            a = np.eye(dimension)[i] * step
            b = np.eye(dimension)[j] * step
            hessian[i, j] = -(
                target(point + a + b)
                - target(point + a - b)
                - target(point - a + b)
                + target(point - a - b)
            ) / (4 * step**2)
    values, vectors = np.linalg.eigh((hessian + hessian.T) / 2)
    # Only the initial proposal uses this local curvature. It does not
    # approximate the posterior target or truncate posterior uncertainty.
    return (vectors / np.maximum(values, 1.0)) @ vectors.T


def parameter_draws(chains, kind):
    output = {
        key: np.empty(chains.shape[:2])
        for key in ["initial", "capacity", "rate_original", "process_sd", "observation_sd"]
    }
    if kind == "ou":
        output["correlation_time_original"] = np.empty(chains.shape[:2])
    for chain in range(chains.shape[0]):
        for draw, z in enumerate(chains[chain]):
            p = physical_parameters(z, kind)
            for key in ["initial", "capacity", "process_sd", "observation_sd"]:
                output[key][chain, draw] = p[key]
            output["rate_original"][chain, draw] = p["rate_per_interval"] / 12
            if kind == "ou":
                output["correlation_time_original"][chain, draw] = 12 / p["decay_per_interval"]
    if kind == "iid":
        del output["process_sd"]
    return output


def predictive_summary(problem, chains, times, observed, count):
    flat = chains.reshape(-1, chains.shape[-1])
    selected = flat[np.linspace(0, len(flat) - 1, min(count, len(flat)), dtype=int)]
    count = len(selected)
    means = np.empty((count, len(times), observed.shape[1]))
    log_means = np.empty_like(means)
    variances = np.empty((count, len(times)))
    measurement = np.empty(count)
    population = np.empty((count, len(times)))
    log_densities = np.empty_like(means)
    joint = np.empty(count)
    for index, z in enumerate(selected):
        mu, latent_cov, noise = problem.conditional_log_distribution(z, times)
        covariance = latent_cov + noise * np.eye(len(times))
        factor = np.linalg.cholesky(covariance)
        variance = covariance.diagonal()
        means[index] = np.exp(mu + 0.5 * variance[:, None])
        log_means[index] = mu
        variances[index] = latent_cov.diagonal()
        measurement[index] = noise
        population[index] = np.exp(log_median(times, z, problem.kind))
        residual = np.log(observed) - mu
        log_densities[index] = -0.5 * (
            np.log(2 * np.pi * variance[:, None]) + residual**2 / variance[:, None]
        ) - np.log(observed)
        if hasattr(problem, "joint_log_prediction"):
            joint[index] = problem.joint_log_prediction(z, times, observed)
        else:
            white = np.linalg.solve(factor, residual)
            joint[index] = (
                -0.5
                * (
                    observed.size * np.log(2 * np.pi)
                    + 2 * observed.shape[1] * np.log(factor.diagonal()).sum()
                    + (white**2).sum()
                )
                - np.log(observed).sum()
            )
    mean = means.mean(0)
    lower, upper = np.empty_like(mean), np.empty_like(mean)
    for i in range(len(times)):
        for j in range(observed.shape[1]):
            total_variance = variances[:, i] + measurement
            lower[i, j] = np.exp(
                gaussian_mixture_quantile(log_means[:, i, j], total_variance, 0.05)
            )
            upper[i, j] = np.exp(
                gaussian_mixture_quantile(log_means[:, i, j], total_variance, 0.95)
            )
    density = logsumexp(log_densities, axis=0) - np.log(count)
    parameter_variance = log_means.var(axis=0, ddof=0)
    process_variance = variances.mean(axis=0)[:, None] * np.ones_like(mean)
    observation_variance = np.full_like(mean, measurement.mean())
    metrics = {
        "rmse": float(np.sqrt(np.mean((observed - mean) ** 2))),
        "coverage": float(((observed >= lower) & (observed <= upper)).mean()),
        "mean_width": float((upper - lower).mean()),
        "marginal_predictive_nll": float(-density.mean()),
        "joint_predictive_nll_per_entry": float(
            -(logsumexp(joint) - np.log(count)) / observed.size
        ),
        "parameter_log_variance": float(parameter_variance.mean()),
        "latent_log_variance": float(process_variance.mean()),
        "measurement_log_variance": float(observation_variance.mean()),
        "posterior_prediction_draws": count,
    }
    output = {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "population_median_lower": np.quantile(population, 0.05, axis=0),
        "population_median_upper": np.quantile(population, 0.95, axis=0),
    }
    return metrics, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["ou", "iid"], default="ou")
    parser.add_argument("--cutoff", type=float, default=168.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=6000)
    parser.add_argument("--draws", type=int, default=12000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--prediction-draws", type=int, default=12000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-chains", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data_path = Path("data/raw/waterKefirTrialsReferece.csv")
    frame, columns = validate_frame(read_trial_csv(data_path))
    training = frame.loc[frame.time <= args.cutoff]
    if len(training) < 2 or training.time.iloc[-1] != args.cutoff:
        raise ValueError("cutoff must be an observed training time.")
    origin = float(frame.time.iloc[0])
    problem = ObservationProblem(
        (training.time.to_numpy() - origin) / 12,
        training[columns].to_numpy(),
        args.kind,
        args.noise_multiplier,
    )
    previous = None
    if args.reuse_chains:
        previous = json.loads((args.output_dir / "experiment.json").read_text())
        if previous["data_sha256"] != hashlib.sha256(data_path.read_bytes()).hexdigest():
            raise ValueError("Cannot reuse posterior samples after input data changed.")
        for path in [
            Path("src/kefir_models/bayesian_observation.py"),
            Path("src/kefir_models/mcmc.py"),
        ]:
            if (
                previous["source_sha256"][str(path)]
                != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError("Cannot reuse posterior samples after target or sampler changes.")
        for key in ["kind", "cutoff", "noise_multiplier", "warmup", "draws", "chains", "seed"]:
            if previous["arguments"][key] != getattr(args, key):
                raise ValueError(f"Cannot reuse posterior samples with changed {key}.")
        chains = np.load(args.output_dir / "posterior_chains.npz")["chains"]
        if chains.shape != (args.chains, args.draws, len(problem.prior.mean)):
            raise ValueError("Saved posterior shape differs from the declared run.")
        accept = previous["acceptance_by_chain"]
        map_objectives = previous["map_objectives"]
    else:
        rng = np.random.default_rng(args.seed)
        modes = []
        for initial in [
            problem.prior.mean,
            problem.prior.mean + rng.normal(0, 0.4, len(problem.prior.mean)),
        ]:
            result = minimize(
                lambda z: -problem.log_posterior(z),
                initial,
                method="BFGS",
                options={"maxiter": 1000, "gtol": 1e-5},
            )
            modes.append(result)
        best = min(modes, key=lambda fit: fit.fun)
        center = best.x
        covariance = curvature_covariance(problem.log_posterior, center)
        factor = np.linalg.cholesky(covariance)
        chains = []
        accept = []
        for index in range(args.chains):
            initial = center + 2 * factor @ rng.normal(size=len(center))
            result = metropolis_chain(
                problem.log_posterior,
                initial,
                covariance,
                args.warmup,
                args.draws,
                args.seed + 1009 * (index + 1),
            )
            chains.append(result.draws)
            accept.append(result.acceptance)
            print(
                f"{args.kind} cutoff={args.cutoff} chain={index + 1} "
                f"acceptance={result.acceptance:.3f}",
                flush=True,
            )
        chains = np.asarray(chains)
        np.savez_compressed(args.output_dir / "posterior_chains.npz", chains=chains)
        map_objectives = [float(mode.fun) for mode in modes]
    physical = parameter_draws(chains, args.kind)
    summaries = []
    for name, values in physical.items():
        summary = {
            "parameter": name,
            "mean": float(values.mean()),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
            **diagnostics(values),
        }
        summaries.append(summary)
    for index in range(chains.shape[-1]):
        summaries.append({"parameter": f"coordinate_{index}", **diagnostics(chains[:, :, index])})
    diagnostics_frame = pd.DataFrame(summaries)
    diagnostics_frame.to_csv(args.output_dir / "posterior_summary.csv", index=False)
    passed = bool(
        (diagnostics_frame.rank_split_rhat < 1.01).all()
        and (diagnostics_frame.bulk_ess >= 400).all()
        and (diagnostics_frame.tail_ess >= 400).all()
    )
    full = args.cutoff == float(frame.time.max())
    evaluation = frame.iloc[1:] if full else frame.loc[frame.time > args.cutoff]
    metrics = []
    for seed in [0]:
        score, prediction = predictive_summary(
            problem,
            chains,
            (evaluation.time.to_numpy() - origin) / 12,
            evaluation[columns].to_numpy(),
            args.prediction_draws,
        )
        metrics.append(
            {
                "kind": args.kind,
                "cutoff": args.cutoff,
                "full_fit": full,
                "noise_multiplier": args.noise_multiplier,
                "prediction_seed": seed,
                "evaluation_count": len(evaluation) * len(columns),
                "computation_passed": passed,
                **score,
            }
        )
        if seed == 0:
            output = pd.DataFrame({"time": evaluation.time.to_numpy()})
            for index, column in enumerate(columns):
                output[f"{column}_observed"] = evaluation[column].to_numpy()
                for key in ["mean", "lower", "upper"]:
                    output[f"{column}_{key}"] = prediction[key][:, index]
            for key in ["population_median_lower", "population_median_upper"]:
                output[key] = prediction[key]
            output.to_csv(args.output_dir / "posterior_predictive.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    metadata = {
        "arguments": vars(args),
        "prior_coordinate_mean": problem.prior.mean.tolist(),
        "prior_coordinate_sd": problem.prior.sd.tolist(),
        "acceptance_by_chain": accept,
        "map_objectives": map_objectives,
        "prediction_method": "Rao-Blackwellized Gaussian-mixture CDF and log densities",
        "sampling_source_sha256": (
            previous.get("sampling_source_sha256") or previous["source_sha256"]
        )
        if previous
        else None,
        "computation_passed": passed,
        "time_unit_original": 12.0,
        "units": "Original numerical units; physical units unconfirmed.",
        "elapsed_seconds": time.monotonic() - start,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path("src/kefir_models/bayesian_observation.py"),
                Path("src/kefir_models/mcmc.py"),
                Path("src/kefir_models/posterior_prediction.py"),
            ]
        },
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(diagnostics_frame.to_string(index=False), flush=True)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
