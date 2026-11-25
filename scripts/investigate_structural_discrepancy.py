"""Fit the predeclared sharing-by-kernel structural discrepancy comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from investigate_bayesian_observation import (
    curvature_covariance,
    parameter_draws,
    predictive_summary,
)
from scipy.optimize import minimize

from kefir_models.mcmc import diagnostics, metropolis_chain
from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.structural_discrepancy import StructuralProblem


def interval_score(observed, lower, upper, alpha=0.1):
    return (upper - lower) + 2 / alpha * (
        (lower - observed) * (observed < lower) + (observed - upper) * (observed > upper)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=["ou", "matern32"], default="ou")
    parser.add_argument("--sharing", choices=["shared", "independent"], default="shared")
    parser.add_argument("--cutoff", type=float, default=168.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
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
        raise ValueError("cutoff must be an observed time.")
    origin = float(frame.time.iloc[0])
    problem = StructuralProblem(
        (training.time.to_numpy() - origin) / 12,
        training[columns].to_numpy(),
        args.kernel,
        args.sharing,
        args.noise_multiplier,
    )
    sources = [
        Path(__file__),
        Path("scripts/investigate_bayesian_observation.py"),
        Path("src/kefir_models/structural_discrepancy.py"),
        Path("src/kefir_models/bayesian_observation.py"),
        Path("src/kefir_models/mcmc.py"),
        Path("src/kefir_models/posterior_prediction.py"),
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if args.reuse_chains:
        previous = json.loads((args.output_dir / "experiment.json").read_text())
        if previous["data_sha256"] != data_hash or previous["source_sha256"] != hashes:
            raise ValueError(
                "Input or experiment sources changed; refit instead of reusing samples."
            )
        for key in ["kernel", "sharing", "cutoff", "noise_multiplier", "draws", "warmup", "seed"]:
            if previous["arguments"][key] != getattr(args, key):
                raise ValueError(f"Cannot reuse samples with changed {key}.")
        chains = np.load(args.output_dir / "posterior_chains.npz")["chains"]
        acceptance = previous["acceptance_by_chain"]
        objectives = previous["map_objectives"]
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
        point = min(modes, key=lambda mode: mode.fun).x
        covariance = curvature_covariance(problem.log_posterior, point)
        factor = np.linalg.cholesky(covariance)
        chains = []
        acceptance = []
        for chain in range(4):
            result = metropolis_chain(
                problem.log_posterior,
                point + 2 * factor @ rng.normal(size=6),
                covariance,
                args.warmup,
                args.draws,
                args.seed + 1009 * (chain + 1),
            )
            chains.append(result.draws)
            acceptance.append(result.acceptance)
            print(
                f"{args.sharing} {args.kernel} cutoff={args.cutoff} chain={chain + 1} "
                f"acceptance={result.acceptance:.3f}",
                flush=True,
            )
        chains = np.asarray(chains)
        objectives = [float(mode.fun) for mode in modes]
        np.savez_compressed(args.output_dir / "posterior_chains.npz", chains=chains)
    parameters = parameter_draws(chains, "ou")
    rows = []
    for name, values in parameters.items():
        rows.append(
            {
                "parameter": name,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
                **diagnostics(values),
            }
        )
    for i in range(6):
        rows.append({"parameter": f"coordinate_{i}", **diagnostics(chains[:, :, i])})
    diag = pd.DataFrame(rows)
    diag.to_csv(args.output_dir / "posterior_summary.csv", index=False)
    passed = bool(
        (diag.rank_split_rhat < 1.01).all()
        and (diag.bulk_ess >= 400).all()
        and (diag.tail_ess >= 400).all()
    )
    full = args.cutoff == float(frame.time.max())
    targets = {"rollout": frame.iloc[1:] if full else frame.loc[frame.time > args.cutoff]}
    if not full:
        targets["one_step"] = targets["rollout"].iloc[:1]
    metrics = []
    for name, evaluation in targets.items():
        observed = evaluation[columns].to_numpy()
        score, prediction = predictive_summary(
            problem,
            chains,
            (evaluation.time.to_numpy() - origin) / 12,
            observed,
            args.prediction_draws,
        )
        rows = {
            "kernel": args.kernel,
            "sharing": args.sharing,
            "cutoff": args.cutoff,
            "noise_multiplier": args.noise_multiplier,
            "full_fit": full,
            "task": name,
            "computation_passed": passed,
            "evaluation_count": observed.size,
            **score,
            "interval_score": float(
                interval_score(observed, prediction["lower"], prediction["upper"]).mean()
            ),
            "mean_bias": float((observed - prediction["mean"]).mean()),
            "above_upper": int((observed > prediction["upper"]).sum()),
            "below_lower": int((observed < prediction["lower"]).sum()),
        }
        metrics.append(rows)
        output = pd.DataFrame({"time": evaluation.time.to_numpy()})
        for j, column in enumerate(columns):
            output[f"{column}_observed"] = observed[:, j]
            for key in ["mean", "lower", "upper"]:
                output[f"{column}_{key}"] = prediction[key][:, j]
        output.to_csv(args.output_dir / f"{name}_predictive.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    metadata = {
        "arguments": vars(args),
        "computation_passed": passed,
        "acceptance_by_chain": acceptance,
        "map_objectives": objectives,
        "prior_coordinate_mean": problem.prior.mean.tolist(),
        "prior_coordinate_sd": problem.prior.sd.tolist(),
        "data_sha256": data_hash,
        "source_sha256": hashes,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "elapsed_seconds": time.monotonic() - start,
        "time_scale_original": 12,
        "correlation_parameter": "Matched e-fold time 12/lambda for both kernels",
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    print(
        diag[["parameter", "rank_split_rhat", "bulk_ess", "tail_ess"]].to_string(index=False),
        flush=True,
    )
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
