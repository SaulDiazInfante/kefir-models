"""Predeclared time-holdout and transition-resolution experiments for REV-07.

All neural candidates use the exact-state observation assumption of the original
model. The classical comparator instead assumes independent additive Gaussian
measurement errors conditional on its fitted deterministic trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kefir_models.forecast_validation import forecast_data, training_and_evaluation
from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.sde_compare import (
    NeuralSDEFunction,
    TransitionBatch,
    build_transition_batch,
    compute_interval_coverage,
    compute_rmse,
    count_trainable_parameters,
    logistic_growth_prediction,
    simulate_sde_paths,
    to_original_scale,
    train_classical_logistic_model,
)
from kefir_models.transition_moments import gaussian_moment_nll, gaussian_transition_moments

# Fixed before evaluating holdouts. No selection or tuning on test observations.
CANDIDATES = [
    ("wide_euler", 32, 2, 1, 1e-4),
    ("wide_moments", 32, 2, 16, 1e-4),
    ("compact_euler", 8, 1, 1, 1e-4),
    ("compact_moments", 8, 1, 16, 1e-4),
    ("compact_moments_regularized", 8, 1, 16, 1e-2),
]


def train_candidate(data, units, layers, steps, decay, epochs, seed):
    torch.manual_seed(seed)
    model = NeuralSDEFunction(units, layers, 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=decay)
    batch = build_transition_batch(data, torch.device("cpu"))
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = gaussian_moment_nll(model, batch, steps)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training objective.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        history.append(float(loss.detach()))
        if (epoch + 1) % 250 == 0:
            print(f"  epoch={epoch + 1} loss={history[-1]:.6f}", flush=True)
    model.eval()
    return model, history


def selected_transitions(evaluation, cutoff, full_fit):
    batch = build_transition_batch(evaluation, torch.device("cpu"))
    if full_fit:
        return batch
    selected = (
        batch.time[:, 0] >= (cutoff - evaluation.scaling.time_min) / evaluation.scaling.time_scale
    )
    return TransitionBatch(
        *(getattr(batch, name)[selected] for name in ("time", "current", "following", "delta_time"))
    )


def summarize_forecast(model, evaluation, cutoff, full_fit, paths, output_dir, name):
    data = evaluation if full_fit else forecast_data(evaluation, cutoff)
    observed = data.raw_frame[data.trial_columns].to_numpy()
    rows = []
    for seed in [8, 9, 10]:
        prediction = simulate_sde_paths(model, data, paths, 0.9, seed, torch.device("cpu"), 0.001)
        mean = to_original_scale(prediction.mean, data).squeeze(-1)
        lower = to_original_scale(prediction.lower, data).squeeze(-1)
        upper = to_original_scale(prediction.upper, data).squeeze(-1)
        variance = (prediction.std.numpy().squeeze(-1) * data.scaling.value_std) ** 2
        variance = np.maximum(variance[1:], 1e-12)
        proxy_nll = 0.5 * (np.log(2 * np.pi * variance) + (observed[1:] - mean[1:]) ** 2 / variance)
        # Exclude the conditioning row: its deterministic zero-width band is
        # guaranteed to cover and must not improve a calibration diagnostic.
        rows.append(
            {
                "simulation_seed": seed,
                "forecast_rmse": compute_rmse(observed[1:], mean[1:]),
                "forecast_coverage": compute_interval_coverage(observed[1:], lower[1:], upper[1:]),
                "forecast_mean_width": float((upper[1:] - lower[1:]).mean()),
                "forecast_gaussian_proxy_nll": float(proxy_nll.mean()),
                "maximum_mean_mc_se": float(prediction.std[1:].max())
                * data.scaling.value_std
                / math.sqrt(paths),
            }
        )
        if seed == 8:
            out = pd.DataFrame({"time": data.raw_frame.time})
            for index, column in enumerate(data.trial_columns):
                out[f"{column}_observed"] = observed[:, index]
                out[f"{column}_mean"] = mean[:, index]
                out[f"{column}_lower"] = lower[:, index]
                out[f"{column}_upper"] = upper[:, index]
            out.to_csv(output_dir / f"{name}_forecast.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-fits", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_fits:
        previous = json.loads((args.output_dir / "experiment.json").read_text())
        for key in ("cutoff", "epochs", "seed"):
            if previous["arguments"][key] != getattr(args, key):
                raise ValueError(f"Cannot reuse fits with a changed {key}.")
    path = Path("data/raw/waterKefirTrialsReferece.csv")
    frame, columns = validate_frame(read_trial_csv(path))
    train, evaluation = training_and_evaluation(frame, columns, args.cutoff)
    full_fit = args.cutoff == float(frame.time.max())
    transitions = selected_transitions(evaluation, args.cutoff, full_fit)
    rows = []
    for name, units, layers, steps, decay in CANDIDATES:
        start = time.monotonic()
        print(f"cutoff={args.cutoff} candidate={name}", flush=True)
        if args.reuse_fits:
            checkpoint = torch.load(args.output_dir / f"{name}.pt", weights_only=False)
            if checkpoint["candidate"] != [name, units, layers, steps, decay]:
                raise ValueError("Saved candidate settings differ from this experiment.")
            model = NeuralSDEFunction(units, layers, 0.001)
            model.load_state_dict(checkpoint["model_state_dict"])
            history = pd.read_csv(args.output_dir / f"{name}_training.csv").nll_normalized.tolist()
        else:
            model, history = train_candidate(
                train, units, layers, steps, decay, args.epochs, args.seed
            )
        with torch.no_grad():
            mean, variance = gaussian_transition_moments(model, transitions, 64)
            nll = 0.5 * (
                torch.log(2 * math.pi * variance)
                + (transitions.following - mean).square() / variance
            )
            physical_nll = float(nll.mean()) + math.log(train.scaling.value_std)
            conditional_rmse = (
                float((transitions.following - mean).square().mean().sqrt())
                * train.scaling.value_std
            )
            covered = (transitions.following - mean).abs() <= 1.6448536269514722 * variance.sqrt()
            conditional_coverage = float(covered.float().mean())
        forecasts = summarize_forecast(
            model, evaluation, args.cutoff, full_fit, args.paths, args.output_dir, name
        )
        common = {
            "model": name,
            "cutoff": args.cutoff,
            "full_fit": full_fit,
            "training_seed": args.seed,
            "epochs": args.epochs,
            "parameter_count": count_trainable_parameters(model),
            "training_substeps": steps,
            "weight_decay": decay,
            "train_final_nll_normalized": history[-1],
            "conditional_gaussian_proxy_nll": physical_nll,
            "conditional_rmse": conditional_rmse,
            "conditional_gaussian_coverage": conditional_coverage,
            "evaluation_count": len(transitions.current),
            "train_value_mean": train.scaling.value_mean,
            "train_value_std": train.scaling.value_std,
            "train_time_scale": train.scaling.time_scale,
            "elapsed_seconds": time.monotonic() - start,
        }
        rows.extend({**common, **forecast} for forecast in forecasts)
        pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
        pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "nll_normalized": history}).to_csv(
            args.output_dir / f"{name}_training.csv", index=False
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaling": train.scaling.__dict__,
                "candidate": [name, units, layers, steps, decay],
            },
            args.output_dir / f"{name}.pt",
        )
        print({**common, **forecasts[0]}, flush=True)
    fit = train_classical_logistic_model(train, 0.05, 3000, torch.device("cpu"))
    observed = evaluation.raw_frame[columns].to_numpy()
    predicted = logistic_growth_prediction(
        evaluation.time,
        torch.tensor(observed[0], dtype=torch.float32),
        torch.tensor(fit.growth_rate),
        torch.tensor(fit.carrying_capacity),
    )
    predicted = predicted.numpy().squeeze(-1)
    train_residual = train.raw_frame[columns].to_numpy() - fit.prediction.numpy().squeeze(-1)
    variance = float((train_residual**2).mean())
    selection = (
        np.arange(1, len(frame))
        if full_fit
        else np.flatnonzero(frame.time.to_numpy() > args.cutoff)
    )
    residual = observed[selection] - predicted[selection]
    baseline = {
        "model": "classical_observation_noise",
        "cutoff": args.cutoff,
        "full_fit": full_fit,
        "training_seed": args.seed,
        "simulation_seed": 0,
        "parameter_count": 3,
        "forecast_rmse": float(np.sqrt((residual**2).mean())),
        "forecast_coverage": float(
            (np.abs(residual) <= 1.6448536269514722 * np.sqrt(variance)).mean()
        ),
        "forecast_mean_width": 2 * 1.6448536269514722 * np.sqrt(variance),
        "forecast_gaussian_proxy_nll": 0.5
        * (math.log(2 * math.pi * variance) + (residual**2).mean() / variance),
        "evaluation_count": residual.size,
    }
    rows.append(baseline)
    baseline_frame = pd.DataFrame({"time": frame.time})
    for index, column in enumerate(columns):
        baseline_frame[f"{column}_observed"] = observed[:, index]
        baseline_frame[f"{column}_mean"] = predicted[:, index]
        baseline_frame[f"{column}_lower"] = predicted[:, index] - 1.6448536269514722 * np.sqrt(
            variance
        )
        baseline_frame[f"{column}_upper"] = predicted[:, index] + 1.6448536269514722 * np.sqrt(
            variance
        )
    baseline_frame.to_csv(args.output_dir / "classical_observation_noise_forecast.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    metadata = {
        "arguments": vars(args),
        "candidates": CANDIDATES,
        "scaling": train.scaling.__dict__,
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "source_sha256": {
            str(source): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in [
                Path(__file__),
                Path("src/kefir_models/transition_moments.py"),
                Path("src/kefir_models/forecast_validation.py"),
            ]
        },
        "observation_models": {
            "neural": "exact observed states, no measurement error",
            "classical": "iid additive Gaussian measurement error",
        },
        "evaluation": (
            "64-step Gaussian quadrature proxy; 3 simulation seeds; excludes conditioning row"
        ),
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
