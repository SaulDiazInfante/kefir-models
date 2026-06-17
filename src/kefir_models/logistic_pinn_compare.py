"""Fit water kefir data with deterministic and stochastic logistic PINNs.

This script solves an inverse problem for the logistic growth equation

    dX / dt = r X (1 - X / K)

using a physics-informed neural network (PINN) trajectory and learnable
parameters ``r`` and ``K``. It also fits a stochastic logistic differential
equation

    dX_t = r X_t (1 - X_t / K) dt + sigma X_t dW_t

where the drift is constrained by the same PINN residual and ``sigma`` is
identified with an Euler-Maruyama transition likelihood. Outputs include fitted
parameters, metrics, training losses, and plots comparing the deterministic
PINN fit with stochastic SDE predictive intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kefir_models.ode_fit import (
    DEFAULT_DATA_PATH,
    read_trial_csv,
    resolve_data_path,
    select_device,
    validate_frame,
)
from kefir_models.plot_style import (
    COMPARISON_FIGURE_SIZE,
    COMPARISON_LAYOUT,
    LOSS_COLORS,
    LOSS_FIGURE_SIZE,
    LOSS_LAYOUT,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_comparison_axis_style,
    deduplicated_legend,
    model_line_style,
    save_fixed_layout_png,
    save_tight_png,
    trial_line_style,
)


@dataclass(frozen=True)
class LogisticPINNData:
    """Observed kefir trial data on original response scale."""

    time_hours: Tensor
    normalized_time: Tensor
    observed: Tensor
    mask: Tensor
    raw_frame: pd.DataFrame
    trial_columns: list[str]
    time_min: float
    time_scale: float
    value_scale: float
    minimum_capacity: float
    initial_level: float
    initial_values: np.ndarray


@dataclass(frozen=True)
class TransitionBatch:
    """Observed one-step transitions for stochastic logistic fitting."""

    current: Tensor
    following: Tensor
    delta_time: Tensor


@dataclass(frozen=True)
class LogisticPINNFit:
    """Trained PINN trajectory and inverse-problem parameters."""

    model: nn.Module
    parameters: nn.Module
    history: pd.DataFrame
    final_loss: float


@dataclass(frozen=True)
class SDESummary:
    """Monte Carlo summary of simulated stochastic logistic trajectories."""

    time: np.ndarray
    mean_by_trial: np.ndarray
    lower_by_trial: np.ndarray
    upper_by_trial: np.ndarray
    std_by_trial: np.ndarray
    population_mean: np.ndarray
    population_lower: np.ndarray
    population_upper: np.ndarray


class LogisticTrajectoryPINN(nn.Module):
    """Positive neural trajectory used in the logistic PINN residual."""

    def __init__(
        self,
        hidden_units: int,
        hidden_layers: int,
        initial_level: float,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")

        layers: list[nn.Module] = [nn.Linear(1, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
        layers.append(nn.Linear(hidden_units, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_output(initial_level)

    def forward(self, normalized_time: Tensor) -> Tensor:
        """Evaluate a positive biomass trajectory."""

        if normalized_time.ndim == 1:
            normalized_time = normalized_time.unsqueeze(-1)
        return F.softplus(self.network(normalized_time)) + 1e-6

    def _initialize_output(self, initial_level: float) -> None:
        """Start from a nearly constant positive trajectory."""

        final_layer = self.network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(
                final_layer.bias,
                inverse_softplus(max(float(initial_level), 1e-6)),
            )


class LogisticParameterization(nn.Module):
    """Positive logistic parameters with a capacity lower bound."""

    def __init__(
        self,
        minimum_capacity: float,
        initial_capacity: float,
        initial_growth_rate: float,
        initial_diffusion: float | None = None,
    ) -> None:
        super().__init__()
        if minimum_capacity <= 0.0:
            raise ValueError("minimum_capacity must be positive.")
        if initial_capacity <= minimum_capacity:
            raise ValueError("initial_capacity must exceed minimum_capacity.")
        if initial_growth_rate <= 0.0:
            raise ValueError("initial_growth_rate must be positive.")

        self.register_buffer(
            "minimum_capacity",
            torch.tensor(float(minimum_capacity), dtype=torch.float32),
        )
        self.raw_growth_rate = nn.Parameter(
            torch.tensor(
                inverse_softplus(initial_growth_rate),
                dtype=torch.float32,
            ),
        )
        self.raw_capacity_margin = nn.Parameter(
            torch.tensor(
                inverse_softplus(initial_capacity - minimum_capacity),
                dtype=torch.float32,
            ),
        )
        if initial_diffusion is None:
            self.raw_diffusion = None
        else:
            if initial_diffusion <= 0.0:
                raise ValueError("initial_diffusion must be positive.")
            self.raw_diffusion = nn.Parameter(
                torch.tensor(
                    inverse_softplus(initial_diffusion),
                    dtype=torch.float32,
                ),
            )

    @property
    def growth_rate(self) -> Tensor:
        """Return positive logistic growth rate."""

        return F.softplus(self.raw_growth_rate) + 1e-8

    @property
    def carrying_capacity(self) -> Tensor:
        """Return carrying capacity above the observed-data lower bound."""

        return self.minimum_capacity + F.softplus(self.raw_capacity_margin) + 1e-8

    @property
    def diffusion(self) -> Tensor:
        """Return positive multiplicative SDE diffusion."""

        if self.raw_diffusion is None:
            raise AttributeError("This parameterization has no diffusion term.")
        return F.softplus(self.raw_diffusion) + 1e-8


def inverse_softplus(value: float) -> float:
    """Return x such that softplus(x) is approximately ``value``."""

    if value <= 0.0:
        raise ValueError("value must be positive.")
    return math.log(math.expm1(value)) if value < 30.0 else value


def load_args_from_json(json_path: Path) -> dict:
    """Load CLI defaults from a JSON file."""

    with json_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if "data_path" in config:
        config["data_path"] = Path(config["data_path"])
    if "output_dir" in config:
        config["output_dir"] = Path(config["output_dir"])
    return config


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Fit deterministic and stochastic logistic PINN inverse models "
            "to water kefir trial data."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON configuration file. Overrides other defaults.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV file with a time column and one or more trial columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logistic_pinn_outputs"),
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--deterministic-epochs",
        type=int,
        default=3000,
        help="Training epochs for the deterministic logistic PINN.",
    )
    parser.add_argument(
        "--stochastic-epochs",
        type=int,
        default=3000,
        help="Training epochs for the stochastic logistic PINN/SDE.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-3,
        help="Adam learning rate for both PINN fits.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=32,
        help="Width of each hidden layer in the PINN trajectory.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=3,
        help="Number of hidden layers in the PINN trajectory.",
    )
    parser.add_argument(
        "--collocation-points",
        type=int,
        default=128,
        help="Number of time points used in the physics residual.",
    )
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=1.0,
        help="Weight for the logistic equation residual.",
    )
    parser.add_argument(
        "--initial-weight",
        type=float,
        default=1.0,
        help="Weight for matching the observed initial values.",
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=0.1,
        help="Weight for the SDE Euler-Maruyama transition likelihood.",
    )
    parser.add_argument(
        "--initial-diffusion",
        type=float,
        default=0.02,
        help="Initial multiplicative diffusion for the stochastic logistic SDE.",
    )
    parser.add_argument(
        "--sde-paths",
        type=int,
        default=5000,
        help="Monte Carlo paths used for stochastic predictive intervals.",
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=0.90,
        help="Central predictive interval level for the SDE band.",
    )
    parser.add_argument(
        "--plot-points",
        type=int,
        default=300,
        help="Number of dense time points used in comparison plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible training and simulation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save diagnostic plots.",
    )

    temp_args, _ = parser.parse_known_args(args)
    if temp_args.config is not None:
        parser.set_defaults(**load_args_from_json(temp_args.config))

    return parser.parse_args(args)


def build_logistic_data(
    frame: pd.DataFrame,
    trial_columns: list[str],
    device: torch.device,
) -> LogisticPINNData:
    """Convert validated trial data into tensors for logistic PINN fitting."""

    time_values = frame["time"].to_numpy(dtype=np.float32)
    observed_with_nan = frame[trial_columns].to_numpy(dtype=np.float32)
    observed_mask = ~np.isnan(observed_with_nan)
    observed_values = observed_with_nan[observed_mask]
    if observed_values.size == 0:
        raise ValueError("No numeric observations were found.")
    if np.nanmin(observed_with_nan) <= 0.0:
        raise ValueError("The logistic PINN requires positive observed values.")

    time_min = float(np.min(time_values))
    time_scale = float(np.max(time_values) - time_min)
    if time_scale <= 0.0:
        time_scale = 1.0

    value_scale = float(np.nanstd(observed_with_nan, ddof=0))
    if value_scale <= 0.0:
        value_scale = 1.0

    maximum_observed = float(np.nanmax(observed_with_nan))
    minimum_capacity = maximum_observed + 1e-3
    initial_values = observed_with_nan[0].astype(np.float32)
    initial_level = float(np.nanmean(initial_values))

    normalized_time = ((time_values - time_min) / time_scale).reshape(-1, 1)
    observed_filled = np.nan_to_num(observed_with_nan, nan=0.0)

    return LogisticPINNData(
        time_hours=torch.tensor(time_values, dtype=torch.float32, device=device),
        normalized_time=torch.tensor(
            normalized_time,
            dtype=torch.float32,
            device=device,
        ),
        observed=torch.tensor(observed_filled, dtype=torch.float32, device=device),
        mask=torch.tensor(observed_mask, dtype=torch.bool, device=device),
        raw_frame=frame,
        trial_columns=trial_columns,
        time_min=time_min,
        time_scale=time_scale,
        value_scale=value_scale,
        minimum_capacity=minimum_capacity,
        initial_level=initial_level,
        initial_values=initial_values,
    )


def estimate_initial_growth_rate(data: LogisticPINNData) -> float:
    """Estimate a stable positive starting value for logistic growth."""

    observed_mean = data.raw_frame[data.trial_columns].mean(axis=1).to_numpy(float)
    initial_value = float(observed_mean[0])
    final_value = float(observed_mean[-1])
    initial_capacity = data.minimum_capacity * 1.08
    elapsed_time = max(float(data.time_hours[-1].detach().cpu()) - data.time_min, 1.0)

    if not (0.0 < initial_value < final_value < initial_capacity):
        return 0.03

    initial_ratio = (initial_capacity - initial_value) / initial_value
    final_ratio = (initial_capacity - final_value) / final_value
    if initial_ratio <= 0.0 or final_ratio <= 0.0:
        return 0.03

    growth_rate = math.log(initial_ratio / final_ratio) / elapsed_time
    return float(np.clip(growth_rate, 1e-4, 1.0))


def make_model_and_parameters(
    data: LogisticPINNData,
    hidden_units: int,
    hidden_layers: int,
    stochastic: bool,
    initial_diffusion: float,
) -> tuple[LogisticTrajectoryPINN, LogisticParameterization]:
    """Initialize a trajectory PINN and positive logistic parameters."""

    initial_capacity = max(data.minimum_capacity * 1.08, data.initial_level + 1.0)
    initial_growth_rate = estimate_initial_growth_rate(data)
    diffusion_start = initial_diffusion if stochastic else None

    model = LogisticTrajectoryPINN(
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        initial_level=data.initial_level,
    )
    parameters = LogisticParameterization(
        minimum_capacity=data.minimum_capacity,
        initial_capacity=initial_capacity,
        initial_growth_rate=initial_growth_rate,
        initial_diffusion=diffusion_start,
    )
    return model, parameters


def data_mse_loss(prediction: Tensor, data: LogisticPINNData) -> Tensor:
    """Return normalized MSE between one population curve and all trials."""

    residual = prediction[:, None] - data.observed
    return torch.mean((residual[data.mask] / data.value_scale).pow(2))


def initial_mse_loss(prediction: Tensor, data: LogisticPINNData) -> Tensor:
    """Return normalized MSE at the first observed time point."""

    first_mask = data.mask[0]
    first_residual = prediction[0] - data.observed[0, first_mask]
    return torch.mean((first_residual / data.value_scale).pow(2))


def physics_residual_loss(
    model: LogisticTrajectoryPINN,
    parameters: LogisticParameterization,
    collocation_time: Tensor,
    data: LogisticPINNData,
) -> Tensor:
    """Return normalized logistic differential-equation residual loss."""

    trajectory = model(collocation_time).squeeze(-1)
    derivative = torch.autograd.grad(
        trajectory.sum(),
        collocation_time,
        create_graph=True,
    )[0].squeeze(-1)
    drift = (
        data.time_scale
        * parameters.growth_rate
        * trajectory
        * (1.0 - trajectory / parameters.carrying_capacity)
    )
    return torch.mean(((derivative - drift) / data.value_scale).pow(2))


def build_transition_batch(data: LogisticPINNData) -> TransitionBatch:
    """Build observed transitions for Euler-Maruyama SDE fitting."""

    current = data.observed[:-1]
    following = data.observed[1:]
    transition_mask = data.mask[:-1] & data.mask[1:]
    delta_time = (data.time_hours[1:] - data.time_hours[:-1])[:, None].expand_as(
        current,
    )

    if not bool(transition_mask.any()):
        raise ValueError("No complete one-step transitions were found.")

    return TransitionBatch(
        current=current[transition_mask].unsqueeze(-1),
        following=following[transition_mask].unsqueeze(-1),
        delta_time=delta_time[transition_mask].unsqueeze(-1),
    )


def transition_negative_log_likelihood(
    parameters: LogisticParameterization,
    transitions: TransitionBatch,
) -> Tensor:
    """Return Gaussian Euler-Maruyama NLL for stochastic logistic transitions."""

    current = transitions.current.clamp_min(1e-6)
    delta_time = transitions.delta_time.clamp_min(1e-8)
    drift = parameters.growth_rate * current * (1.0 - current / parameters.carrying_capacity)
    mean = current + drift * delta_time
    variance = (parameters.diffusion * current).pow(2) * delta_time + 1e-6
    squared_error = (transitions.following - mean).pow(2)
    return 0.5 * (torch.log(2.0 * math.pi * variance) + squared_error / variance).mean()


def train_logistic_pinn(
    data: LogisticPINNData,
    hidden_units: int,
    hidden_layers: int,
    epochs: int,
    learning_rate: float,
    collocation_points: int,
    physics_weight: float,
    initial_weight: float,
    stochastic: bool,
    transition_weight: float,
    initial_diffusion: float,
    device: torch.device,
) -> LogisticPINNFit:
    """Train deterministic or stochastic logistic PINN inverse model."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if collocation_points < 2:
        raise ValueError("collocation_points must be at least 2.")

    model, parameters = make_model_and_parameters(
        data=data,
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        stochastic=stochastic,
        initial_diffusion=initial_diffusion,
    )
    model = model.to(device)
    parameters = parameters.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(parameters.parameters()),
        lr=learning_rate,
    )
    collocation_time = torch.linspace(
        0.0,
        1.0,
        collocation_points,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    collocation_time.requires_grad_(True)
    transitions = build_transition_batch(data) if stochastic else None
    history_rows: list[dict[str, float]] = []
    label = "stochastic" if stochastic else "deterministic"

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data.normalized_time).squeeze(-1)
        data_loss = data_mse_loss(prediction, data)
        initial_loss = initial_mse_loss(prediction, data)
        physics_loss = physics_residual_loss(
            model=model,
            parameters=parameters,
            collocation_time=collocation_time,
            data=data,
        )
        loss = data_loss + initial_weight * initial_loss + physics_weight * physics_loss

        transition_nll = torch.tensor(float("nan"), device=device)
        if stochastic:
            if transitions is None:
                raise RuntimeError("Missing stochastic transition batch.")
            transition_nll = transition_negative_log_likelihood(parameters, transitions)
            loss = loss + transition_weight * transition_nll

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(parameters.parameters()),
            max_norm=25.0,
        )
        optimizer.step()

        row = {
            "epoch": float(epoch),
            "total_loss": float(loss.detach().cpu()),
            "data_mse": float(data_loss.detach().cpu()),
            "initial_mse": float(initial_loss.detach().cpu()),
            "physics_mse": float(physics_loss.detach().cpu()),
            "transition_nll": float(transition_nll.detach().cpu()),
            "growth_rate": float(parameters.growth_rate.detach().cpu()),
            "carrying_capacity": float(parameters.carrying_capacity.detach().cpu()),
        }
        if stochastic:
            row["diffusion"] = float(parameters.diffusion.detach().cpu())
        history_rows.append(row)

        if epoch == 1 or epoch % 250 == 0 or epoch == epochs:
            if stochastic:
                print(
                    f"{label}_epoch={epoch:05d} "
                    f"loss={row['total_loss']:.6f} "
                    f"nll={row['transition_nll']:.6f} "
                    f"r={row['growth_rate']:.6f} "
                    f"K={row['carrying_capacity']:.6f} "
                    f"sigma={row['diffusion']:.6f}",
                )
            else:
                print(
                    f"{label}_epoch={epoch:05d} "
                    f"loss={row['total_loss']:.6f} "
                    f"r={row['growth_rate']:.6f} "
                    f"K={row['carrying_capacity']:.6f}",
                )

    model.eval()
    parameters.eval()
    history = pd.DataFrame(history_rows)
    return LogisticPINNFit(
        model=model,
        parameters=parameters,
        history=history,
        final_loss=float(history["total_loss"].iloc[-1]),
    )


def evaluate_pinn(
    fit: LogisticPINNFit,
    time_hours: np.ndarray,
    data: LogisticPINNData,
    device: torch.device,
) -> np.ndarray:
    """Evaluate a trained PINN at original-scale time values."""

    normalized_time = (time_hours.astype(np.float32) - data.time_min) / data.time_scale
    time_tensor = torch.tensor(
        normalized_time.reshape(-1, 1),
        dtype=torch.float32,
        device=device,
    )
    fit.model.eval()
    with torch.no_grad():
        prediction = fit.model(time_tensor).squeeze(-1)
    return prediction.detach().cpu().numpy()


def simulate_logistic_sde(
    time_hours: np.ndarray,
    initial_values: np.ndarray,
    growth_rate: float,
    carrying_capacity: float,
    diffusion: float,
    paths: int,
    interval_level: float,
    seed: int,
    device: torch.device,
) -> SDESummary:
    """Simulate stochastic logistic SDE paths and summarize intervals."""

    if paths < 1:
        raise ValueError("paths must be at least 1.")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must be between 0 and 1.")

    time_tensor = torch.tensor(time_hours, dtype=torch.float32, device=device)
    initial_tensor = torch.tensor(initial_values, dtype=torch.float32, device=device)
    current = initial_tensor.unsqueeze(0).repeat(paths, 1).clamp_min(1e-6)
    trajectories = [current]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    growth = torch.tensor(growth_rate, dtype=torch.float32, device=device)
    capacity = torch.tensor(carrying_capacity, dtype=torch.float32, device=device)
    sigma = torch.tensor(diffusion, dtype=torch.float32, device=device)

    for index in range(len(time_tensor) - 1):
        delta_time = (time_tensor[index + 1] - time_tensor[index]).clamp_min(1e-8)
        noise = torch.randn(current.shape, generator=generator, device=device)
        drift = growth * current * (1.0 - current / capacity)
        diffusion_term = sigma * current
        current = (
            current + drift * delta_time + diffusion_term * torch.sqrt(delta_time) * noise
        ).clamp_min(1e-6)
        trajectories.append(current)

    trajectory_tensor = torch.stack(trajectories).detach().cpu()
    alpha = (1.0 - interval_level) / 2.0
    mean_by_trial = trajectory_tensor.mean(dim=1).numpy()
    lower_by_trial = torch.quantile(trajectory_tensor, alpha, dim=1).numpy()
    upper_by_trial = torch.quantile(trajectory_tensor, 1.0 - alpha, dim=1).numpy()
    std_by_trial = trajectory_tensor.std(dim=1, unbiased=False).numpy()

    flattened = trajectory_tensor.reshape(len(time_hours), -1)
    return SDESummary(
        time=time_hours,
        mean_by_trial=mean_by_trial,
        lower_by_trial=lower_by_trial,
        upper_by_trial=upper_by_trial,
        std_by_trial=std_by_trial,
        population_mean=flattened.mean(dim=1).numpy(),
        population_lower=torch.quantile(flattened, alpha, dim=1).numpy(),
        population_upper=torch.quantile(flattened, 1.0 - alpha, dim=1).numpy(),
    )


def observed_fitted_pairs(
    observed: np.ndarray,
    fitted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed/fitted vectors after dropping missing observations."""

    mask = ~np.isnan(observed)
    return observed[mask], fitted[mask]


def compute_rmse(observed: np.ndarray, fitted: np.ndarray) -> float:
    """Compute RMSE over observed entries only."""

    observed_values, fitted_values = observed_fitted_pairs(observed, fitted)
    return float(np.sqrt(np.mean((observed_values - fitted_values) ** 2)))


def compute_r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    """Compute R-squared over observed entries only."""

    observed_values, fitted_values = observed_fitted_pairs(observed, fitted)
    residual_sum_squares = float(np.sum((observed_values - fitted_values) ** 2))
    total_sum_squares = float(
        np.sum((observed_values - np.mean(observed_values)) ** 2),
    )
    if total_sum_squares == 0.0:
        return 1.0 if residual_sum_squares == 0.0 else float("nan")
    return 1.0 - residual_sum_squares / total_sum_squares


def compute_interval_coverage(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute empirical predictive-interval coverage."""

    mask = ~np.isnan(observed)
    covered = (observed[mask] >= lower[mask]) & (observed[mask] <= upper[mask])
    return float(np.mean(covered))


def count_trainable_parameters(modules: Iterable[nn.Module]) -> int:
    """Count trainable parameters across modules."""

    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def build_comparison_frames(
    data: LogisticPINNData,
    deterministic_fit: LogisticPINNFit,
    stochastic_fit: LogisticPINNFit,
    observed_sde: SDESummary,
    dense_sde: SDESummary,
    plot_points: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build observed-grid and dense-grid comparison frames."""

    observed_time = data.raw_frame["time"].to_numpy(dtype=np.float32)
    dense_time = np.linspace(
        float(observed_time.min()),
        float(observed_time.max()),
        plot_points,
        dtype=np.float32,
    )
    deterministic_observed = evaluate_pinn(
        deterministic_fit,
        observed_time,
        data,
        device,
    )
    stochastic_observed = evaluate_pinn(
        stochastic_fit,
        observed_time,
        data,
        device,
    )
    deterministic_dense = evaluate_pinn(deterministic_fit, dense_time, data, device)
    stochastic_dense = evaluate_pinn(stochastic_fit, dense_time, data, device)

    observed_frame = pd.DataFrame({"time": observed_time})
    for index, column in enumerate(data.trial_columns):
        observed_frame[f"{column}_observed"] = data.raw_frame[column].to_numpy()
        observed_frame[f"{column}_deterministic_pinn"] = deterministic_observed
        observed_frame[f"{column}_stochastic_pinn"] = stochastic_observed
        observed_frame[f"{column}_sde_mean"] = observed_sde.mean_by_trial[:, index]
        observed_frame[f"{column}_sde_lower"] = observed_sde.lower_by_trial[:, index]
        observed_frame[f"{column}_sde_upper"] = observed_sde.upper_by_trial[:, index]
        observed_frame[f"{column}_sde_std"] = observed_sde.std_by_trial[:, index]

    observed_frame["observed_mean"] = data.raw_frame[data.trial_columns].mean(axis=1)
    observed_frame["deterministic_pinn"] = deterministic_observed
    observed_frame["stochastic_pinn_drift"] = stochastic_observed
    observed_frame["sde_mean"] = observed_sde.population_mean
    observed_frame["sde_lower"] = observed_sde.population_lower
    observed_frame["sde_upper"] = observed_sde.population_upper

    dense_frame = pd.DataFrame(
        {
            "time": dense_time,
            "deterministic_pinn": deterministic_dense,
            "stochastic_pinn_drift": stochastic_dense,
            "sde_mean": dense_sde.population_mean,
            "sde_lower": dense_sde.population_lower,
            "sde_upper": dense_sde.population_upper,
        },
    )
    return observed_frame, dense_frame


def build_metrics_frame(
    data: LogisticPINNData,
    comparison_frame: pd.DataFrame,
    deterministic_fit: LogisticPINNFit,
    stochastic_fit: LogisticPINNFit,
) -> pd.DataFrame:
    """Build model-comparison metrics on the observed time grid."""

    observed = data.raw_frame[data.trial_columns].to_numpy(dtype=float)
    deterministic_curve = comparison_frame["deterministic_pinn"].to_numpy()
    stochastic_curve = comparison_frame["stochastic_pinn_drift"].to_numpy()
    deterministic_fitted = np.repeat(
        deterministic_curve[:, None],
        len(data.trial_columns),
        axis=1,
    )
    stochastic_fitted = np.repeat(
        stochastic_curve[:, None],
        len(data.trial_columns),
        axis=1,
    )
    sde_mean = comparison_frame[[f"{column}_sde_mean" for column in data.trial_columns]].to_numpy()
    sde_lower = comparison_frame[
        [f"{column}_sde_lower" for column in data.trial_columns]
    ].to_numpy()
    sde_upper = comparison_frame[
        [f"{column}_sde_upper" for column in data.trial_columns]
    ].to_numpy()

    rows = [
        {
            "model": "Deterministic Logistic PINN",
            "fitted": deterministic_fitted,
            "parameter_count": count_trainable_parameters(
                [deterministic_fit.model, deterministic_fit.parameters],
            ),
            "final_objective": deterministic_fit.final_loss,
            "growth_rate": float(
                deterministic_fit.parameters.growth_rate.detach().cpu(),
            ),
            "carrying_capacity": float(
                deterministic_fit.parameters.carrying_capacity.detach().cpu(),
            ),
            "diffusion": np.nan,
            "interval_coverage": np.nan,
            "mean_interval_width": np.nan,
        },
        {
            "model": "Stochastic Logistic PINN Drift",
            "fitted": stochastic_fitted,
            "parameter_count": count_trainable_parameters(
                [stochastic_fit.model, stochastic_fit.parameters],
            ),
            "final_objective": stochastic_fit.final_loss,
            "growth_rate": float(stochastic_fit.parameters.growth_rate.detach().cpu()),
            "carrying_capacity": float(
                stochastic_fit.parameters.carrying_capacity.detach().cpu(),
            ),
            "diffusion": float(stochastic_fit.parameters.diffusion.detach().cpu()),
            "interval_coverage": np.nan,
            "mean_interval_width": np.nan,
        },
        {
            "model": "Stochastic Logistic SDE Mean",
            "fitted": sde_mean,
            "parameter_count": count_trainable_parameters(
                [stochastic_fit.model, stochastic_fit.parameters],
            ),
            "final_objective": stochastic_fit.final_loss,
            "growth_rate": float(stochastic_fit.parameters.growth_rate.detach().cpu()),
            "carrying_capacity": float(
                stochastic_fit.parameters.carrying_capacity.detach().cpu(),
            ),
            "diffusion": float(stochastic_fit.parameters.diffusion.detach().cpu()),
            "interval_coverage": compute_interval_coverage(
                observed,
                sde_lower,
                sde_upper,
            ),
            "mean_interval_width": float(np.nanmean(sde_upper - sde_lower)),
        },
    ]

    metric_rows = []
    for row in rows:
        fitted = row.pop("fitted")
        metric_rows.append(
            {
                **row,
                "rmse": compute_rmse(observed, fitted),
                "r_squared": compute_r_squared(observed, fitted),
                "observation_count": int(np.sum(~np.isnan(observed))),
            },
        )
    return pd.DataFrame(metric_rows)


def build_parameter_frame(
    deterministic_fit: LogisticPINNFit,
    stochastic_fit: LogisticPINNFit,
) -> pd.DataFrame:
    """Build a long-form parameter table."""

    return pd.DataFrame(
        [
            {
                "model": "deterministic_logistic_pinn",
                "parameter": "growth_rate",
                "value": float(
                    deterministic_fit.parameters.growth_rate.detach().cpu(),
                ),
            },
            {
                "model": "deterministic_logistic_pinn",
                "parameter": "carrying_capacity",
                "value": float(
                    deterministic_fit.parameters.carrying_capacity.detach().cpu(),
                ),
            },
            {
                "model": "stochastic_logistic_sde",
                "parameter": "growth_rate",
                "value": float(stochastic_fit.parameters.growth_rate.detach().cpu()),
            },
            {
                "model": "stochastic_logistic_sde",
                "parameter": "carrying_capacity",
                "value": float(
                    stochastic_fit.parameters.carrying_capacity.detach().cpu(),
                ),
            },
            {
                "model": "stochastic_logistic_sde",
                "parameter": "diffusion",
                "value": float(stochastic_fit.parameters.diffusion.detach().cpu()),
            },
        ],
    )


def save_comparison_plot(
    observed_frame: pd.DataFrame,
    dense_frame: pd.DataFrame,
    trial_columns: list[str],
    interval_level: float,
    path: Path,
) -> None:
    """Save observed data, deterministic PINN, and stochastic SDE comparison."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axis = plt.subplots(figsize=COMPARISON_FIGURE_SIZE)
    fig.subplots_adjust(**COMPARISON_LAYOUT)

    y_series = [
        pd.to_numeric(observed_frame[f"{column}_observed"], errors="coerce").dropna()
        for column in trial_columns
        if f"{column}_observed" in observed_frame.columns
    ]
    for column in ["observed_mean"]:
        if column in observed_frame.columns:
            y_series.append(
                pd.to_numeric(observed_frame[column], errors="coerce").dropna(),
            )
    for column in ["deterministic_pinn", "sde_mean", "sde_lower", "sde_upper"]:
        if column in dense_frame.columns:
            y_series.append(
                pd.to_numeric(dense_frame[column], errors="coerce").dropna(),
            )
    y_min = min(float(series.min()) for series in y_series if not series.empty)
    y_max = max(float(series.max()) for series in y_series if not series.empty)
    y_range = y_max - y_min if y_max > y_min else 1.0
    axis.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    x_values = [
        pd.to_numeric(observed_frame["time"], errors="coerce").dropna(),
        pd.to_numeric(dense_frame["time"], errors="coerce").dropna(),
    ]
    x_min = min(float(series.min()) for series in x_values if not series.empty)
    x_max = max(float(series.max()) for series in x_values if not series.empty)
    x_range = x_max - x_min if x_max > x_min else 1.0
    axis.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)

    for index, column in enumerate(trial_columns):
        axis.plot(
            observed_frame["time"],
            observed_frame[f"{column}_observed"],
            **trial_line_style(index),
            label=column.replace("_", " ").title(),
        )

    axis.plot(
        observed_frame["time"],
        observed_frame["observed_mean"],
        **model_line_style("observed_mean"),
        label=MODEL_LABELS["observed_mean"],
    )
    axis.plot(
        dense_frame["time"],
        dense_frame["deterministic_pinn"],
        **model_line_style("deterministic_pinn"),
        label=MODEL_LABELS["deterministic_pinn"],
    )
    axis.plot(
        dense_frame["time"],
        dense_frame["sde_mean"],
        **model_line_style("logistic_pinn_sde"),
        label=MODEL_LABELS["logistic_pinn_sde"],
    )
    axis.fill_between(
        dense_frame["time"],
        dense_frame["sde_lower"],
        dense_frame["sde_upper"],
        color=MODEL_COLORS["logistic_pinn_sde_band"],
        alpha=0.14,
        label=f"Logistic\nSDE PINN\n{interval_level:.0%} C.B.",
    )
    axis.plot(
        dense_frame["time"],
        dense_frame["sde_lower"],
        color=MODEL_COLORS["logistic_pinn_sde_band"],
        linewidth=0.9,
        alpha=0.8,
    )
    axis.plot(
        dense_frame["time"],
        dense_frame["sde_upper"],
        color=MODEL_COLORS["logistic_pinn_sde_band"],
        linewidth=0.9,
        alpha=0.8,
    )

    axis.set_title("Water Kefir Logistic PINN: Deterministic vs Stochastic")
    apply_comparison_axis_style(axis)
    deduplicated_legend(axis)
    save_fixed_layout_png(fig, path)
    plt.close(fig)


def save_loss_plot(
    deterministic_history: pd.DataFrame,
    stochastic_history: pd.DataFrame,
    path: Path,
) -> None:
    """Save deterministic and stochastic training objective histories."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(1, 2, figsize=LOSS_FIGURE_SIZE)
    fig.subplots_adjust(**LOSS_LAYOUT)
    for axis, history, title in [
        (axes[0], deterministic_history, "Deterministic PINN"),
        (axes[1], stochastic_history, "Stochastic PINN/SDE"),
    ]:
        axis.plot(
            history["epoch"],
            history["total_loss"],
            color=LOSS_COLORS["total_loss"],
            linewidth=1.6,
            label="Total",
        )
        axis.plot(
            history["epoch"],
            history["data_mse"],
            color=LOSS_COLORS["data_mse"],
            linewidth=1.6,
            label="Data",
        )
        axis.plot(
            history["epoch"],
            history["physics_mse"],
            color=LOSS_COLORS["physics_mse"],
            linewidth=1.6,
            label="Physics",
        )
        if "transition_nll" in history.columns and history["transition_nll"].notna().any():
            axis.plot(
                history["epoch"],
                history["transition_nll"],
                color=LOSS_COLORS["transition_nll"],
                linewidth=1.6,
                label="Transition NLL",
            )
        plotted_columns = ["total_loss", "data_mse", "physics_mse"]
        if "transition_nll" in history.columns and history["transition_nll"].notna().any():
            plotted_columns.append("transition_nll")
        plotted_values = history[plotted_columns].to_numpy(dtype=float)
        if np.isfinite(plotted_values).all() and (plotted_values > 0.0).all():
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Objective")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    save_tight_png(fig, path)
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    data: LogisticPINNData,
    deterministic_fit: LogisticPINNFit,
    stochastic_fit: LogisticPINNFit,
    comparison_frame: pd.DataFrame,
    dense_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    parameter_frame: pd.DataFrame,
    interval_level: float,
    plot: bool,
) -> None:
    """Persist tables, checkpoints, and plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "water_kefir_logistic_pinn_comparison.csv"
    dense_path = output_dir / "water_kefir_logistic_pinn_dense.csv"
    metrics_path = output_dir / "water_kefir_logistic_pinn_metrics.csv"
    parameters_path = output_dir / "water_kefir_logistic_pinn_parameters.csv"
    deterministic_history_path = output_dir / "deterministic_logistic_pinn_training_loss.csv"
    stochastic_history_path = output_dir / "stochastic_logistic_pinn_training_loss.csv"
    deterministic_model_path = output_dir / "deterministic_logistic_pinn_model.pt"
    stochastic_model_path = output_dir / "stochastic_logistic_pinn_model.pt"
    comparison_plot_path = output_dir / "water_kefir_logistic_pinn_comparison.png"
    loss_plot_path = output_dir / "water_kefir_logistic_pinn_losses.png"

    comparison_frame.to_csv(comparison_path, index=False)
    dense_frame.to_csv(dense_path, index=False)
    metrics_frame.to_csv(metrics_path, index=False)
    parameter_frame.to_csv(parameters_path, index=False)
    deterministic_fit.history.to_csv(deterministic_history_path, index=False)
    stochastic_fit.history.to_csv(stochastic_history_path, index=False)

    torch.save(
        {
            "model_state_dict": deterministic_fit.model.state_dict(),
            "parameter_state_dict": deterministic_fit.parameters.state_dict(),
            "trial_columns": data.trial_columns,
            "time_min": data.time_min,
            "time_scale": data.time_scale,
            "value_scale": data.value_scale,
            "model_type": "deterministic_logistic_pinn",
        },
        deterministic_model_path,
    )
    torch.save(
        {
            "model_state_dict": stochastic_fit.model.state_dict(),
            "parameter_state_dict": stochastic_fit.parameters.state_dict(),
            "trial_columns": data.trial_columns,
            "time_min": data.time_min,
            "time_scale": data.time_scale,
            "value_scale": data.value_scale,
            "model_type": "stochastic_logistic_pinn_sde",
        },
        stochastic_model_path,
    )

    if plot:
        save_comparison_plot(
            observed_frame=comparison_frame,
            dense_frame=dense_frame,
            trial_columns=data.trial_columns,
            interval_level=interval_level,
            path=comparison_plot_path,
        )
        save_loss_plot(
            deterministic_history=deterministic_fit.history,
            stochastic_history=stochastic_fit.history,
            path=loss_plot_path,
        )

    print(metrics_frame.to_string(index=False))
    print(f"Saved comparison data to {comparison_path}")
    print(f"Saved dense plot data to {dense_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved parameters to {parameters_path}")
    print(f"Saved deterministic history to {deterministic_history_path}")
    print(f"Saved stochastic history to {stochastic_history_path}")
    print(f"Saved deterministic checkpoint to {deterministic_model_path}")
    print(f"Saved stochastic checkpoint to {stochastic_model_path}")
    if plot:
        print(f"Saved comparison plot to {comparison_plot_path}")
        print(f"Saved loss plot to {loss_plot_path}")


def main() -> None:
    """Run deterministic and stochastic logistic PINN comparison."""

    args = parse_args()
    if args.deterministic_epochs < 1:
        raise ValueError("deterministic_epochs must be at least 1.")
    if args.stochastic_epochs < 1:
        raise ValueError("stochastic_epochs must be at least 1.")
    if args.plot_points < 2:
        raise ValueError("plot_points must be at least 2.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    data_path = resolve_data_path(args.data_path)
    frame = read_trial_csv(data_path)
    frame, trial_columns = validate_frame(frame)
    data = build_logistic_data(frame, trial_columns, device)

    print("Training deterministic logistic PINN...")
    deterministic_fit = train_logistic_pinn(
        data=data,
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        epochs=args.deterministic_epochs,
        learning_rate=args.learning_rate,
        collocation_points=args.collocation_points,
        physics_weight=args.physics_weight,
        initial_weight=args.initial_weight,
        stochastic=False,
        transition_weight=args.transition_weight,
        initial_diffusion=args.initial_diffusion,
        device=device,
    )

    print("Training stochastic logistic PINN/SDE...")
    stochastic_fit = train_logistic_pinn(
        data=data,
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        epochs=args.stochastic_epochs,
        learning_rate=args.learning_rate,
        collocation_points=args.collocation_points,
        physics_weight=args.physics_weight,
        initial_weight=args.initial_weight,
        stochastic=True,
        transition_weight=args.transition_weight,
        initial_diffusion=args.initial_diffusion,
        device=device,
    )

    observed_time = data.raw_frame["time"].to_numpy(dtype=np.float32)
    dense_time = np.linspace(
        float(observed_time.min()),
        float(observed_time.max()),
        args.plot_points,
        dtype=np.float32,
    )
    stochastic_growth_rate = float(stochastic_fit.parameters.growth_rate.detach().cpu())
    stochastic_capacity = float(
        stochastic_fit.parameters.carrying_capacity.detach().cpu(),
    )
    stochastic_diffusion = float(stochastic_fit.parameters.diffusion.detach().cpu())

    observed_sde = simulate_logistic_sde(
        time_hours=observed_time,
        initial_values=data.initial_values,
        growth_rate=stochastic_growth_rate,
        carrying_capacity=stochastic_capacity,
        diffusion=stochastic_diffusion,
        paths=args.sde_paths,
        interval_level=args.interval_level,
        seed=args.seed + 1,
        device=device,
    )
    dense_sde = simulate_logistic_sde(
        time_hours=dense_time,
        initial_values=data.initial_values,
        growth_rate=stochastic_growth_rate,
        carrying_capacity=stochastic_capacity,
        diffusion=stochastic_diffusion,
        paths=args.sde_paths,
        interval_level=args.interval_level,
        seed=args.seed + 2,
        device=device,
    )
    comparison_frame, dense_frame = build_comparison_frames(
        data=data,
        deterministic_fit=deterministic_fit,
        stochastic_fit=stochastic_fit,
        observed_sde=observed_sde,
        dense_sde=dense_sde,
        plot_points=args.plot_points,
        device=device,
    )
    metrics_frame = build_metrics_frame(
        data=data,
        comparison_frame=comparison_frame,
        deterministic_fit=deterministic_fit,
        stochastic_fit=stochastic_fit,
    )
    parameter_frame = build_parameter_frame(
        deterministic_fit=deterministic_fit,
        stochastic_fit=stochastic_fit,
    )

    save_outputs(
        output_dir=args.output_dir,
        data=data,
        deterministic_fit=deterministic_fit,
        stochastic_fit=stochastic_fit,
        comparison_frame=comparison_frame,
        dense_frame=dense_frame,
        metrics_frame=metrics_frame,
        parameter_frame=parameter_frame,
        interval_level=args.interval_level,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
