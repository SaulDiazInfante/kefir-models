"""Compare a frozen Neural SDE's Euler transitions with refined conditional simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kefir_models.ode_fit import build_training_data, read_trial_csv, validate_frame
from kefir_models.sde_compare import NeuralSDEFunction, build_transition_batch
from kefir_models.transition_moments import gaussian_transition_moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/neural_sde_comparison_outputs/water_kefir_neural_sde_model.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/rev07/frozen"))
    parser.add_argument("--paths", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    frame, columns = validate_frame(read_trial_csv(Path("data/raw/waterKefirTrialsReferece.csv")))
    data = build_training_data(frame, columns)
    batch = build_transition_batch(data, torch.device("cpu"))
    saved = torch.load(args.checkpoint, weights_only=False)
    if saved.get("scaling") != data.scaling.__dict__:
        raise ValueError("This frozen diagnostic requires a checkpoint fitted on the full data.")
    candidate = saved.get("candidate", ["legacy", 32, 2, 1, 1e-4])
    model = NeuralSDEFunction(candidate[1], candidate[2], 0.001)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    states = batch.current.clone().requires_grad_()
    derivative = torch.autograd.grad(model.drift(batch.time, states).sum(), states)[0]
    amplification = (derivative * batch.delta_time).detach().numpy().ravel()
    rows = []
    with torch.no_grad():
        euler_mean = batch.current + model.drift(batch.time, batch.current) * batch.delta_time
        for seed in [8, 9, 10]:
            for steps in [1, 16, 64]:
                generator = torch.Generator().manual_seed(seed)
                current = batch.current[None].repeat(args.paths, 1, 1)
                step = batch.delta_time[None] / steps
                for index in range(steps):
                    time = (batch.time[None] + index * step).expand_as(current)
                    current = (
                        current
                        + model.drift(time, current) * step
                        + model.diffusion(time, current)
                        * step.sqrt()
                        * torch.randn(current.shape, generator=generator)
                    )
                mean = current.mean(0)
                variance = current.var(0, unbiased=False)
                quantiles = torch.quantile(current, torch.tensor([0.05, 0.95]), dim=0)
                proxy_mean, proxy_variance = gaussian_transition_moments(model, batch, steps)
                rows.append(
                    {
                        "seed": seed,
                        "substeps": steps,
                        "paths": args.paths,
                        "conditional_rmse": float((mean - batch.following).square().mean().sqrt())
                        * data.scaling.value_std,
                        "mean_shift_from_euler_rms": float(
                            (mean - euler_mean).square().mean().sqrt()
                        )
                        * data.scaling.value_std,
                        "gaussian_mc_proxy_nll": float(
                            (
                                0.5
                                * (
                                    torch.log(2 * math.pi * variance)
                                    + (batch.following - mean).square() / variance
                                )
                            ).mean()
                        )
                        + math.log(data.scaling.value_std),
                        "coverage": float(
                            ((quantiles[0] <= batch.following) & (batch.following <= quantiles[1]))
                            .float()
                            .mean()
                        ),
                        "moment_closure_mean_error_rms": float(
                            (mean - proxy_mean).square().mean().sqrt()
                        )
                        * data.scaling.value_std,
                        "moment_closure_variance_error_mean": float(
                            (variance - proxy_variance).abs().mean()
                        )
                        * data.scaling.value_std**2,
                        "maximum_mean_mc_se": float(variance.sqrt().max())
                        * data.scaling.value_std
                        / math.sqrt(args.paths),
                    }
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "transitions.csv", index=False)
    diagnostics = {
        "checkpoint": str(args.checkpoint),
        "drift_derivative_times_gap_quantiles": np.quantile(
            amplification, [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "negative_drift_euler_unstable_count": int((amplification < -2).sum()),
        "transition_count": len(amplification),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "note": (
            "Local frozen linearization, not a global nonlinear stability theorem; "
            "MC NLL is a Gaussian moment proxy."
        ),
    }
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(diagnostics, flush=True)


if __name__ == "__main__":
    main()
