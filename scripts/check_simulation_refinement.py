"""Check saved point-estimate simulations and capacity-constraint sensitivity."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kefir_models import logistic_pinn_compare as pinn
from kefir_models import sde_compare as neural
from kefir_models.ode_fit import build_training_data, read_trial_csv, validate_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    parser.add_argument("--paths", type=int, default=3000)
    parser.add_argument("--capacity-sensitivity", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    frame, columns = validate_frame(read_trial_csv(Path("data/raw/waterKefirTrialsReferece.csv")))
    data = build_training_data(frame, columns)
    observed = frame[columns].to_numpy()
    neural_dir = args.results_dir / "neural_sde_comparison_outputs"
    checkpoint = torch.load(
        neural_dir / "water_kefir_neural_sde_model.pt", map_location="cpu", weights_only=False
    )
    model = neural.NeuralSDEFunction(32, 2, checkpoint["min_diffusion"])
    model.load_state_dict(checkpoint["model_state_dict"])
    parameters = pd.read_csv(
        args.results_dir / "logistic_pinn_outputs" / "water_kefir_logistic_pinn_parameters.csv"
    )
    physical = parameters.query("model == 'stochastic_logistic_sde'").set_index("parameter").value
    rows = []
    for seed in [8, 9, 10]:
        for step in [0.002, 0.001, 0.0005]:
            prediction = neural.simulate_sde_paths(model, data, args.paths, 0.9, seed, device, step)
            mean = neural.to_original_scale(prediction.mean, data).squeeze(-1)
            lower = neural.to_original_scale(prediction.lower, data).squeeze(-1)
            upper = neural.to_original_scale(prediction.upper, data).squeeze(-1)
            se = prediction.std.numpy().squeeze(-1) * data.scaling.value_std / np.sqrt(args.paths)
            rows.append(
                {
                    "model": "Neural SDE",
                    "seed": seed,
                    "max_step": step,
                    "rmse": neural.compute_rmse(observed, mean),
                    "coverage": neural.compute_interval_coverage(observed, lower, upper),
                    "mean_width": float((upper - lower).mean()),
                    "maximum_mean_mc_se": float(se.max()),
                    "clipped_values": 0,
                }
            )
        for step in [0.1, 0.05, 0.025]:
            prediction = pinn.simulate_logistic_sde(
                frame.time.to_numpy(),
                frame[columns].iloc[0].to_numpy(),
                physical.growth_rate,
                physical.carrying_capacity,
                physical.diffusion,
                args.paths,
                0.9,
                seed,
                device,
                step,
            )
            rows.append(
                {
                    "model": "Logistic SDE",
                    "seed": seed,
                    "max_step": step,
                    "rmse": pinn.compute_rmse(observed, prediction.mean_by_trial),
                    "coverage": pinn.compute_interval_coverage(
                        observed, prediction.lower_by_trial, prediction.upper_by_trial
                    ),
                    "mean_width": float(
                        (prediction.upper_by_trial - prediction.lower_by_trial).mean()
                    ),
                    "maximum_mean_mc_se": float(
                        prediction.std_by_trial.max() / np.sqrt(args.paths)
                    ),
                    "clipped_values": prediction.clipped_values,
                }
            )
        print(f"Finished simulation refinement seed {seed}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "simulation_refinement.csv", index=False)
    print(
        summary.groupby(["model", "max_step"])[["rmse", "coverage", "mean_width"]]
        .agg(["mean", "std"])
        .to_string()
    )
    if args.capacity_sensitivity:
        data_pinn = pinn.build_logistic_data(frame, columns, device)
        rows = []
        for bound in [data_pinn.minimum_capacity, float(observed.max()) + 0.001]:
            changed = replace(data_pinn, minimum_capacity=bound)
            torch.manual_seed(7)
            for stochastic in [False, True]:
                fit = pinn.train_logistic_pinn(
                    changed, 32, 3, 3000, 0.005, 128, 1.0, 1.0, stochastic, 0.1, 0.02, device
                )
                prediction = pinn.evaluate_pinn(fit, frame.time.to_numpy(), changed, device)
                rows.append(
                    {
                        "stochastic": stochastic,
                        "capacity_lower_bound": bound,
                        "capacity": float(fit.parameters.carrying_capacity.detach()),
                        "growth_rate": float(fit.parameters.growth_rate.detach()),
                        "rmse": float(np.sqrt(np.mean((observed - prediction[:, None]) ** 2))),
                        "final_objective": fit.final_loss,
                    }
                )
        pd.DataFrame(rows).to_csv(args.output_dir / "capacity_sensitivity.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
