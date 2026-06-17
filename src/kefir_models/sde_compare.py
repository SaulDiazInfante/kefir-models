"""Compare classical, deterministic Neural ODE, and stochastic Neural SDE fits.

The Neural SDE uses the Ito form

    dY_t = mu_theta(t, Y_t) dt + sigma_theta(t, Y_t) dW_t

where both the drift and diffusion are neural networks. The SDE is trained with
an Euler-Maruyama Gaussian transition likelihood, then simulated forward to
estimate predictive means and uncertainty intervals.

The classical baseline is a shared-parameter logistic growth ODE fit to all
trials. All models are compared with RMSE, R-squared, AIC, and BIC computed
from observed residuals on the original data scale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kefir_models.ode_fit import (
    DEFAULT_DATA_PATH,
    NeuralODEFunction,
    TrainingData,
    build_training_data,
    inverse_transform_values,
    read_trial_csv,
    resolve_data_path,
    select_device,
    train_model,
    validate_frame,
)
from kefir_models.plot_style import (
    COMPARISON_FIGURE_SIZE,
    COMPARISON_LAYOUT,
    MODEL_COLORS,
    MODEL_LABELS,
    OBJECTIVE_FIGURE_SIZE,
    OBJECTIVE_LAYOUT,
    REFERENCE_LINE_COLOR,
    apply_comparison_axis_style,
    deduplicated_legend,
    model_line_style,
    save_fixed_layout_png,
    save_tight_png,
    trial_line_style,
)


@dataclass(frozen=True)
class TransitionBatch:
    """Observed one-step transitions used by the SDE likelihood."""

    time: Tensor
    current: Tensor
    following: Tensor
    delta_time: Tensor


@dataclass(frozen=True)
class SDEPrediction:
    """Monte Carlo summary of Neural SDE simulated trajectories."""

    trajectories: Tensor
    mean: Tensor
    lower: Tensor
    upper: Tensor
    std: Tensor
    population_mean: Tensor
    population_lower: Tensor
    population_upper: Tensor


@dataclass(frozen=True)
class ClassicalLogisticFit:
    """Fitted two-parameter logistic growth baseline."""

    growth_rate: float
    carrying_capacity: float
    prediction: Tensor
    history: list[float]


class NeuralSDEFunction(nn.Module):
    """Neural drift and diffusion for a one-dimensional Ito SDE."""

    def __init__(
        self,
        hidden_units: int,
        hidden_layers: int,
        min_diffusion: float,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")
        if min_diffusion <= 0:
            raise ValueError("min_diffusion must be positive.")

        self.min_diffusion = min_diffusion
        self.drift_network = build_mlp(hidden_units, hidden_layers)
        self.diffusion_network = build_mlp(hidden_units, hidden_layers)
        self._initialize_diffusion()

    def drift(self, t: Tensor, y: Tensor) -> Tensor:
        """Return the drift mu_theta(t, y)."""

        return self.drift_network(self._features(t, y))

    def diffusion(self, t: Tensor, y: Tensor) -> Tensor:
        """Return a strictly positive diffusion sigma_theta(t, y)."""

        raw_diffusion = self.diffusion_network(self._features(t, y))
        return F.softplus(raw_diffusion) + self.min_diffusion

    def _features(self, t: Tensor, y: Tensor) -> Tensor:
        """Build network inputs from time and state tensors."""

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if t.ndim == 0:
            time_column = torch.ones_like(y[..., :1]) * t
        else:
            time_column = t.reshape(y.shape[:-1] + (1,)).to(y.device)

        return torch.cat((time_column, y), dim=-1)

    def _initialize_diffusion(self) -> None:
        """Start with moderate process noise before likelihood training."""

        final_layer = self.diffusion_network[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.constant_(final_layer.bias, -2.0)


def build_mlp(hidden_units: int, hidden_layers: int) -> nn.Sequential:
    """Build a small fully connected network for drift or diffusion."""

    layers: list[nn.Module] = [nn.Linear(2, hidden_units), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
    layers.append(nn.Linear(hidden_units, 1))
    return nn.Sequential(*layers)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compare Neural ODE and Neural SDE fits for water kefir data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON configuration file. Overrides defaults.",
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
        default=Path("neural_sde_comparison_outputs"),
        help="Directory where comparison outputs are written.",
    )
    parser.add_argument(
        "--ode-epochs",
        type=int,
        default=1000,
        help="Training epochs for the deterministic Neural ODE.",
    )
    parser.add_argument(
        "--sde-epochs",
        type=int,
        default=1000,
        help="Training epochs for the Neural SDE transition likelihood.",
    )
    parser.add_argument(
        "--classical-epochs",
        type=int,
        default=3000,
        help="Training epochs for the classical logistic growth baseline.",
    )
    parser.add_argument(
        "--ode-learning-rate",
        type=float,
        default=1e-2,
        help="Adam learning rate for the Neural ODE.",
    )
    parser.add_argument(
        "--sde-learning-rate",
        type=float,
        default=5e-3,
        help="Adam learning rate for the Neural SDE.",
    )
    parser.add_argument(
        "--classical-learning-rate",
        type=float,
        default=5e-2,
        help="Adam learning rate for the classical logistic growth baseline.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=32,
        help="Width of each hidden layer.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=2,
        help="Number of hidden layers.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam L2 regularization for both models.",
    )
    parser.add_argument(
        "--min-diffusion",
        type=float,
        default=1e-3,
        help="Positive lower bound for the SDE diffusion.",
    )
    parser.add_argument(
        "--ode-method",
        choices=("dopri5", "rk4"),
        default="rk4",
        help="ODE solver used for the deterministic Neural ODE.",
    )
    parser.add_argument(
        "--sde-paths",
        type=int,
        default=1000,
        help="Monte Carlo paths used to summarize the trained Neural SDE.",
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=0.90,
        help="Central predictive interval level for the SDE comparison.",
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
        help="Whether to save a comparison plot.",
    )
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        with open(temp_args.config) as f:
            config = json.load(f)
        if "data_path" in config:
            config["data_path"] = Path(config["data_path"])
        if "output_dir" in config:
            config["output_dir"] = Path(config["output_dir"])
        parser.set_defaults(**config)

    return parser.parse_args()


def build_transition_batch(data: TrainingData, device: torch.device) -> TransitionBatch:
    """Build observed transitions for Euler-Maruyama likelihood training."""

    times = data.time.to(device)
    values = data.values.to(device)
    mask = data.mask.to(device).squeeze(-1)

    current_values = values[:-1].squeeze(-1)
    following_values = values[1:].squeeze(-1)
    transition_mask = mask[:-1] & mask[1:]

    current_times = times[:-1, None].expand_as(current_values)
    delta_times = (times[1:] - times[:-1])[:, None].expand_as(current_values)

    return TransitionBatch(
        time=current_times[transition_mask].unsqueeze(-1),
        current=current_values[transition_mask].unsqueeze(-1),
        following=following_values[transition_mask].unsqueeze(-1),
        delta_time=delta_times[transition_mask].unsqueeze(-1),
    )


def transition_negative_log_likelihood(
    model: NeuralSDEFunction,
    batch: TransitionBatch,
) -> Tensor:
    """Gaussian transition NLL implied by one Euler-Maruyama step."""

    drift = model.drift(batch.time, batch.current)
    diffusion = model.diffusion(batch.time, batch.current)
    mean = batch.current + drift * batch.delta_time
    variance = diffusion.pow(2) * batch.delta_time.clamp_min(1e-8) + 1e-6

    squared_error = (batch.following - mean).pow(2)
    return 0.5 * (torch.log(2.0 * math.pi * variance) + squared_error / variance).mean()


def train_sde_model(
    data: TrainingData,
    hidden_units: int,
    hidden_layers: int,
    learning_rate: float,
    weight_decay: float,
    min_diffusion: float,
    epochs: int,
    device: torch.device,
) -> tuple[NeuralSDEFunction, list[float]]:
    """Train the Neural SDE drift and diffusion networks."""

    model = NeuralSDEFunction(
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        min_diffusion=min_diffusion,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    batch = build_transition_batch(data, device)
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = transition_negative_log_likelihood(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if epoch == 1 or epoch % 250 == 0 or epoch == epochs:
            print(f"sde_epoch={epoch:05d} nll={loss_value:.6f}")

    return model, history


def inverse_softplus(value: float) -> float:
    """Return x such that softplus(x) is approximately value."""

    if value <= 0.0:
        raise ValueError("value must be positive.")
    return math.log(math.expm1(value))


def logistic_growth_prediction(
    time_points: Tensor,
    initial_values: Tensor,
    growth_rate: Tensor,
    carrying_capacity: Tensor,
) -> Tensor:
    """Evaluate the exact logistic growth solution for each trial."""

    initial_values = initial_values.clamp_min(1e-8)
    capacity_margin = (carrying_capacity - initial_values).clamp_min(1e-8)
    initial_ratio = capacity_margin / initial_values
    denominator = 1.0 + initial_ratio.unsqueeze(0) * torch.exp(
        -growth_rate * time_points[:, None],
    )
    return (carrying_capacity / denominator).unsqueeze(-1)


def train_classical_logistic_model(
    data: TrainingData,
    learning_rate: float,
    epochs: int,
    device: torch.device,
) -> ClassicalLogisticFit:
    """Fit a shared logistic growth rate and carrying capacity."""

    observed_array = data.raw_frame[data.trial_columns].to_numpy(dtype=np.float32)
    observed_mask = ~np.isnan(observed_array)
    if not observed_mask.any():
        raise ValueError("No observed values were found for classical fitting.")
    if np.nanmin(observed_array) <= 0.0:
        raise ValueError(
            "The classical logistic baseline requires positive observed values.",
        )

    target = torch.tensor(observed_array[..., None], dtype=torch.float32, device=device)
    mask = torch.tensor(observed_mask[..., None], dtype=torch.bool, device=device)
    time_points = data.time.to(device)
    initial_values = target[0, :, 0]

    maximum_initial_value = float(np.max(observed_array[0]))
    minimum_capacity = maximum_initial_value + 1e-3
    initial_capacity = max(
        float(np.nanmax(observed_array)) * 1.05,
        minimum_capacity + 1.0,
    )
    raw_growth = nn.Parameter(
        torch.tensor(inverse_softplus(3.0), dtype=torch.float32, device=device),
    )
    raw_capacity_margin = nn.Parameter(
        torch.tensor(
            inverse_softplus(initial_capacity - minimum_capacity),
            dtype=torch.float32,
            device=device,
        ),
    )
    optimizer = torch.optim.Adam(
        [raw_growth, raw_capacity_margin],
        lr=learning_rate,
    )
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        growth_rate = F.softplus(raw_growth) + 1e-6
        carrying_capacity = minimum_capacity + F.softplus(raw_capacity_margin) + 1e-6
        prediction = logistic_growth_prediction(
            time_points=time_points,
            initial_values=initial_values,
            growth_rate=growth_rate,
            carrying_capacity=carrying_capacity,
        )
        loss = torch.mean((prediction[mask] - target[mask]).pow(2))
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if epoch == 1 or epoch % 250 == 0 or epoch == epochs:
            print(f"classical_epoch={epoch:05d} mse={loss_value:.6f}")

    with torch.no_grad():
        growth_rate = F.softplus(raw_growth) + 1e-6
        carrying_capacity = minimum_capacity + F.softplus(raw_capacity_margin) + 1e-6
        prediction = logistic_growth_prediction(
            time_points=time_points,
            initial_values=initial_values,
            growth_rate=growth_rate,
            carrying_capacity=carrying_capacity,
        )

    return ClassicalLogisticFit(
        growth_rate=float(growth_rate.detach().cpu()),
        carrying_capacity=float(carrying_capacity.detach().cpu()),
        prediction=prediction.detach().cpu(),
        history=history,
    )


def simulate_sde_paths(
    model: NeuralSDEFunction,
    data: TrainingData,
    paths: int,
    interval_level: float,
    seed: int,
    device: torch.device,
) -> SDEPrediction:
    """Simulate Neural SDE paths and summarize predictive uncertainty."""

    if paths < 1:
        raise ValueError("paths must be at least 1.")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must be between 0 and 1.")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    time_points = data.time.to(device)
    initial_state = data.values[0].to(device)
    current = initial_state.unsqueeze(0).repeat(paths, 1, 1)
    trajectories = [current]

    model.eval()
    with torch.no_grad():
        for index in range(len(time_points) - 1):
            time = time_points[index]
            delta_time = time_points[index + 1] - time
            noise = torch.randn(
                current.shape,
                device=device,
                generator=generator,
            )
            current = (
                current
                + model.drift(time, current) * delta_time
                + model.diffusion(time, current) * torch.sqrt(delta_time.clamp_min(1e-8)) * noise
            )
            trajectories.append(current)

    trajectory_tensor = torch.stack(trajectories).detach().cpu()
    alpha = (1.0 - interval_level) / 2.0
    lower_quantile = alpha
    upper_quantile = 1.0 - alpha

    mean = trajectory_tensor.mean(dim=1)
    lower = torch.quantile(trajectory_tensor, lower_quantile, dim=1)
    upper = torch.quantile(trajectory_tensor, upper_quantile, dim=1)
    std = trajectory_tensor.std(dim=1, unbiased=False)

    flattened = trajectory_tensor.squeeze(-1).reshape(len(time_points), -1)
    population_mean = flattened.mean(dim=1)
    population_lower = torch.quantile(flattened, lower_quantile, dim=1)
    population_upper = torch.quantile(flattened, upper_quantile, dim=1)

    return SDEPrediction(
        trajectories=trajectory_tensor,
        mean=mean,
        lower=lower,
        upper=upper,
        std=std,
        population_mean=population_mean,
        population_lower=population_lower,
        population_upper=population_upper,
    )


def to_original_scale(values: Tensor, data: TrainingData) -> np.ndarray:
    """Convert normalized tensor values to the original measurement scale."""

    return inverse_transform_values(values.detach().cpu().numpy(), data.scaling)


def population_curve_matrix(
    values: np.ndarray,
    trial_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trial-shaped fitted values and their population curve."""

    if values.ndim == 1:
        population_curve = values
        return (
            np.repeat(population_curve[:, None], trial_count, axis=1),
            population_curve,
        )

    if values.ndim == 2 and values.shape[1] == 1:
        population_curve = values[:, 0]
        return (
            np.repeat(population_curve[:, None], trial_count, axis=1),
            population_curve,
        )

    if values.ndim != 2 or values.shape[1] != trial_count:
        raise ValueError(
            "Fitted values must be one population curve or one curve per trial.",
        )

    return values, values.mean(axis=1)


def compute_rmse(observed: np.ndarray, fitted: np.ndarray) -> float:
    """Compute RMSE for observed entries only."""

    mask = ~np.isnan(observed)
    residuals = observed[mask] - fitted[mask]
    return float(np.sqrt(np.mean(residuals**2)))


def observed_fitted_pairs(
    observed: np.ndarray,
    fitted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed/fitted vectors after dropping missing observations."""

    mask = ~np.isnan(observed)
    return observed[mask], fitted[mask]


def compute_r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    """Compute R-squared for observed entries only."""

    observed_values, fitted_values = observed_fitted_pairs(observed, fitted)
    residual_sum_squares = float(np.sum((observed_values - fitted_values) ** 2))
    total_sum_squares = float(
        np.sum((observed_values - np.mean(observed_values)) ** 2),
    )
    if total_sum_squares == 0.0:
        return 1.0 if residual_sum_squares == 0.0 else float("nan")
    return 1.0 - residual_sum_squares / total_sum_squares


def gaussian_information_criteria(
    observed: np.ndarray,
    fitted: np.ndarray,
    parameter_count: int,
) -> tuple[float, float, float, int]:
    """Compute Gaussian residual sigma, AIC, BIC, and sample size.

    The supplied parameter count excludes residual variance; the information
    criteria add one fitted variance parameter.
    """

    observed_values, fitted_values = observed_fitted_pairs(observed, fitted)
    residuals = observed_values - fitted_values
    observation_count = int(residuals.size)
    if observation_count == 0:
        return float("nan"), float("nan"), float("nan"), 0

    residual_sum_squares = float(np.sum(residuals**2))
    variance = max(residual_sum_squares / observation_count, np.finfo(float).tiny)
    log_likelihood = -0.5 * observation_count * (math.log(2.0 * math.pi) + 1.0 + math.log(variance))
    likelihood_parameter_count = parameter_count + 1
    aic = 2.0 * likelihood_parameter_count - 2.0 * log_likelihood
    bic = math.log(observation_count) * likelihood_parameter_count - 2.0 * log_likelihood
    return math.sqrt(variance), aic, bic, observation_count


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_interval_coverage(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute empirical coverage of predictive intervals."""

    mask = ~np.isnan(observed)
    covered = (observed[mask] >= lower[mask]) & (observed[mask] <= upper[mask])
    return float(np.mean(covered))


def build_comparison_frame(
    data: TrainingData,
    ode_prediction: Tensor,
    sde_prediction: SDEPrediction,
    classical_fit: ClassicalLogisticFit,
) -> pd.DataFrame:
    """Build a wide data frame comparing observed, neural, and classical fits."""

    ode_values, ode_population_curve = population_curve_matrix(
        to_original_scale(ode_prediction.squeeze(-1), data),
        len(data.trial_columns),
    )
    sde_mean = to_original_scale(sde_prediction.mean.squeeze(-1), data)
    sde_lower = to_original_scale(sde_prediction.lower.squeeze(-1), data)
    sde_upper = to_original_scale(sde_prediction.upper.squeeze(-1), data)
    sde_std = sde_prediction.std.squeeze(-1).numpy() * data.scaling.value_std
    classical_values = classical_fit.prediction.squeeze(-1).numpy()

    output = pd.DataFrame({"time": data.raw_frame["time"].to_numpy()})
    for index, column in enumerate(data.trial_columns):
        output[f"{column}_observed"] = data.raw_frame[column].to_numpy()
        output[f"{column}_ode_fitted"] = ode_values[:, index]
        output[f"{column}_sde_mean"] = sde_mean[:, index]
        output[f"{column}_sde_lower"] = sde_lower[:, index]
        output[f"{column}_sde_upper"] = sde_upper[:, index]
        output[f"{column}_sde_std"] = sde_std[:, index]
        output[f"{column}_classical_logistic_fitted"] = classical_values[:, index]

    output["observed_mean"] = data.raw_frame[data.trial_columns].mean(axis=1)
    output["ode_fitted"] = ode_population_curve
    output["ode_fitted_mean"] = ode_population_curve
    output["sde_mean"] = inverse_transform_values(
        sde_prediction.population_mean.numpy(),
        data.scaling,
    )
    output["classical_logistic_fitted_mean"] = classical_values.mean(axis=1)
    output["sde_lower"] = inverse_transform_values(
        sde_prediction.population_lower.numpy(),
        data.scaling,
    )
    output["sde_upper"] = inverse_transform_values(
        sde_prediction.population_upper.numpy(),
        data.scaling,
    )
    return output


def build_metrics_frame(
    data: TrainingData,
    comparison_frame: pd.DataFrame,
    ode_model: NeuralODEFunction,
    ode_history: list[float],
    sde_model: NeuralSDEFunction,
    sde_history: list[float],
    classical_fit: ClassicalLogisticFit,
) -> pd.DataFrame:
    """Build model comparison metrics."""

    observed = data.raw_frame[data.trial_columns].to_numpy(dtype=float)
    ode_fitted = comparison_frame[
        [f"{column}_ode_fitted" for column in data.trial_columns]
    ].to_numpy()
    sde_mean = comparison_frame[[f"{column}_sde_mean" for column in data.trial_columns]].to_numpy()
    sde_lower = comparison_frame[
        [f"{column}_sde_lower" for column in data.trial_columns]
    ].to_numpy()
    sde_upper = comparison_frame[
        [f"{column}_sde_upper" for column in data.trial_columns]
    ].to_numpy()
    classical_fitted = comparison_frame[
        [f"{column}_classical_logistic_fitted" for column in data.trial_columns]
    ].to_numpy()

    model_specs = [
        {
            "model": "Classical Logistic ODE",
            "fitted": classical_fitted,
            "parameter_count": 2,
            "final_objective": classical_fit.history[-1],
            "objective_type": "original_scale_mse",
            "interval_coverage": np.nan,
            "mean_interval_width": np.nan,
        },
        {
            "model": "Neural ODE",
            "fitted": ode_fitted,
            "parameter_count": count_trainable_parameters(ode_model),
            "final_objective": ode_history[-1],
            "objective_type": "normalized_mse",
            "interval_coverage": np.nan,
            "mean_interval_width": np.nan,
        },
        {
            "model": "Neural SDE",
            "fitted": sde_mean,
            "parameter_count": count_trainable_parameters(sde_model),
            "final_objective": sde_history[-1],
            "objective_type": "transition_nll",
            "interval_coverage": compute_interval_coverage(
                observed,
                sde_lower,
                sde_upper,
            ),
            "mean_interval_width": float(np.nanmean(sde_upper - sde_lower)),
        },
    ]

    rows = []
    for spec in model_specs:
        residual_sigma, aic, bic, observation_count = gaussian_information_criteria(
            observed,
            spec["fitted"],
            int(spec["parameter_count"]),
        )
        rows.append(
            {
                "model": spec["model"],
                "rmse": compute_rmse(observed, spec["fitted"]),
                "r_squared": compute_r_squared(observed, spec["fitted"]),
                "aic": aic,
                "bic": bic,
                "residual_sigma": residual_sigma,
                "observation_count": observation_count,
                "parameter_count": spec["parameter_count"],
                "likelihood_parameter_count": int(spec["parameter_count"]) + 1,
                "final_objective": spec["final_objective"],
                "objective_type": spec["objective_type"],
                "interval_coverage": spec["interval_coverage"],
                "mean_interval_width": spec["mean_interval_width"],
            },
        )

    return pd.DataFrame(rows)


def save_comparison_plot(
    comparison_frame: pd.DataFrame,
    trial_columns: list[str],
    interval_level: float,
    path: Path,
) -> None:
    """Save observed data, ODE fit, and SDE predictive interval plot."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axis = plt.subplots(figsize=COMPARISON_FIGURE_SIZE)
    fig.subplots_adjust(**COMPARISON_LAYOUT)
    time = comparison_frame["time"]

    y_series = [
        pd.to_numeric(comparison_frame[f"{column}_observed"], errors="coerce").dropna()
        for column in trial_columns
        if f"{column}_observed" in comparison_frame.columns
    ]
    for column in [
        "observed_mean",
        "classical_logistic_fitted_mean",
        "ode_fitted_mean",
        "sde_mean",
        "sde_lower",
        "sde_upper",
    ]:
        if column in comparison_frame.columns:
            y_series.append(
                pd.to_numeric(comparison_frame[column], errors="coerce").dropna(),
            )
    y_min = min(float(series.min()) for series in y_series if not series.empty)
    y_max = max(float(series.max()) for series in y_series if not series.empty)
    y_range = y_max - y_min if y_max > y_min else 1.0
    axis.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    x_min, x_max = time.min(), time.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    axis.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)

    for index, column in enumerate(trial_columns):
        axis.plot(
            time,
            comparison_frame[f"{column}_observed"],
            **trial_line_style(index),
            label=column.replace("_", " ").title(),
        )

    axis.plot(
        time,
        comparison_frame["observed_mean"],
        **model_line_style("observed_mean"),
        label=MODEL_LABELS["observed_mean"],
    )
    axis.plot(
        time,
        comparison_frame["ode_fitted_mean"],
        **model_line_style("neural_ode"),
        label=MODEL_LABELS["neural_ode"],
    )
    axis.plot(
        time,
        comparison_frame["sde_mean"],
        **model_line_style("neural_sde"),
        label=MODEL_LABELS["neural_sde"],
    )
    axis.plot(
        time,
        comparison_frame["classical_logistic_fitted_mean"],
        **model_line_style("classical_logistic"),
        label=MODEL_LABELS["classical_logistic"],
    )
    axis.fill_between(
        time,
        comparison_frame["sde_lower"],
        comparison_frame["sde_upper"],
        color=MODEL_COLORS["neural_sde_band"],
        alpha=0.15,
        label=f"NSDE\n{interval_level:.0%} C.B.",
    )
    axis.plot(
        time,
        comparison_frame["sde_lower"],
        color=MODEL_COLORS["neural_sde_band"],
        linewidth=0.9,
        alpha=0.75,
    )
    axis.plot(
        time,
        comparison_frame["sde_upper"],
        color=MODEL_COLORS["neural_sde_band"],
        linewidth=0.9,
        alpha=0.75,
    )

    axis.set_title("Water Kefir: Neural ODE vs Neural SDE vs Logistic ODE")
    apply_comparison_axis_style(axis)
    deduplicated_legend(axis)
    save_fixed_layout_png(fig, path)
    plt.close(fig)


def build_training_objective_scales_frame(
    data: TrainingData,
    classical_history: list[float],
    ode_history: list[float],
    sde_history: list[float],
) -> pd.DataFrame:
    """Return training histories on normalized and original response scales."""

    value_std = data.scaling.value_std
    if value_std <= 0.0:
        raise ValueError("The response standard deviation must be positive.")

    value_variance = value_std**2
    rows: list[dict[str, object]] = []

    for epoch, loss in enumerate(classical_history, start=1):
        rows.append(
            {
                "model": "Classical logistic ODE",
                "objective_type": "mse",
                "epoch": epoch,
                "normalized_value": loss / value_variance,
                "original_scale_value": loss,
            },
        )

    for epoch, loss in enumerate(ode_history, start=1):
        rows.append(
            {
                "model": "Neural ODE",
                "objective_type": "mse",
                "epoch": epoch,
                "normalized_value": loss,
                "original_scale_value": loss * value_variance,
            },
        )

    for epoch, loss in enumerate(sde_history, start=1):
        rows.append(
            {
                "model": "Neural SDE",
                "objective_type": "transition_nll",
                "epoch": epoch,
                "normalized_value": loss,
                # If y = mean + s_y z, then -log p_y(y) = -log p_z(z) + log(s_y).
                "original_scale_value": loss + math.log(value_std),
            },
        )

    return pd.DataFrame(rows)


def save_training_objective_scale_plot(
    history_frame: pd.DataFrame,
    path: Path,
) -> None:
    """Save training histories grouped by normalized and original scales."""

    required_columns = {
        "model",
        "objective_type",
        "epoch",
        "normalized_value",
        "original_scale_value",
    }
    missing_columns = required_columns.difference(history_frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing history columns: {missing}")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    colors = {
        "Classical logistic ODE": MODEL_COLORS["classical_logistic"],
        "Neural ODE": MODEL_COLORS["neural_ode"],
        "Neural SDE": MODEL_COLORS["neural_sde"],
    }
    panel_specs = [
        {
            "title": "Normalized training scale",
            "value_column": "normalized_value",
            "models": ["Classical logistic ODE", "Neural ODE", "Neural SDE"],
            "ylabel": "Objective value",
            "yscale": "symlog",
        },
        {
            "title": "Original response MSE",
            "value_column": "original_scale_value",
            "models": ["Classical logistic ODE", "Neural ODE"],
            "ylabel": "MSE",
            "yscale": "log",
        },
        {
            "title": "Original response transition NLL",
            "value_column": "original_scale_value",
            "models": ["Neural SDE"],
            "ylabel": "NLL",
            "yscale": "log",
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=OBJECTIVE_FIGURE_SIZE)
    fig.subplots_adjust(**OBJECTIVE_LAYOUT)
    for axis, spec in zip(axes, panel_specs):
        for model in spec["models"]:
            model_frame = history_frame[history_frame["model"] == model]
            axis.plot(
                model_frame["epoch"],
                model_frame[spec["value_column"]],
                color=colors[model],
                linewidth=2.0,
                label=model,
            )

        plotted_values = pd.to_numeric(
            history_frame[history_frame["model"].isin(spec["models"])][spec["value_column"]],
            errors="coerce",
        )
        if spec["yscale"] == "log" and (plotted_values > 0.0).all():
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=1e-2)
            axis.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=0.8, alpha=0.55)

        axis.set_title(spec["title"])
        axis.set_xlabel("Epoch")
        axis.set_ylabel(spec["ylabel"])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle("Training Objective Histories by Scale", y=1.02)
    save_tight_png(fig, path)
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    data: TrainingData,
    comparison_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    ode_model: NeuralODEFunction,
    ode_history: list[float],
    sde_model: NeuralSDEFunction,
    sde_history: list[float],
    classical_fit: ClassicalLogisticFit,
    interval_level: float,
    plot: bool,
) -> None:
    """Persist comparison outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "water_kefir_neural_dynamics_comparison.csv"
    metrics_path = output_dir / "model_comparison_metrics.csv"
    ode_history_path = output_dir / "neural_ode_training_loss.csv"
    ode_model_path = output_dir / "water_kefir_neural_ode_model.pt"
    sde_history_path = output_dir / "neural_sde_training_loss.csv"
    sde_model_path = output_dir / "water_kefir_neural_sde_model.pt"
    classical_history_path = output_dir / "classical_logistic_training_loss.csv"
    classical_parameters_path = output_dir / "classical_logistic_parameters.csv"
    plot_path = output_dir / "water_kefir_neural_dynamics_comparison.png"
    objective_scales_path = output_dir / "training_objective_scales.csv"
    objective_plot_path = output_dir / "training_objectives.png"
    objective_scales_frame = build_training_objective_scales_frame(
        data=data,
        classical_history=classical_fit.history,
        ode_history=ode_history,
        sde_history=sde_history,
    )

    comparison_frame.to_csv(comparison_path, index=False)
    metrics_frame.to_csv(metrics_path, index=False)
    objective_scales_frame.to_csv(objective_scales_path, index=False)
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(ode_history) + 1),
            "normalized_mse": ode_history,
        },
    ).to_csv(ode_history_path, index=False)
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(sde_history) + 1),
            "transition_nll": sde_history,
        },
    ).to_csv(sde_history_path, index=False)
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(classical_fit.history) + 1),
            "original_scale_mse": classical_fit.history,
        },
    ).to_csv(classical_history_path, index=False)
    pd.DataFrame(
        [
            {
                "parameter": "growth_rate",
                "value": classical_fit.growth_rate,
            },
            {
                "parameter": "carrying_capacity",
                "value": classical_fit.carrying_capacity,
            },
        ],
    ).to_csv(classical_parameters_path, index=False)
    torch.save(
        {
            "model_state_dict": ode_model.state_dict(),
            "trial_columns": data.trial_columns,
            "scaling": data.scaling.__dict__,
        },
        ode_model_path,
    )
    torch.save(
        {
            "model_state_dict": sde_model.state_dict(),
            "trial_columns": data.trial_columns,
            "scaling": data.scaling.__dict__,
            "min_diffusion": sde_model.min_diffusion,
        },
        sde_model_path,
    )

    if plot:
        save_comparison_plot(
            comparison_frame,
            data.trial_columns,
            interval_level,
            plot_path,
        )
        save_training_objective_scale_plot(
            objective_scales_frame,
            objective_plot_path,
        )

    print(metrics_frame.to_string(index=False))
    print(f"Saved comparison data to {comparison_path}")
    print(f"Saved comparison metrics to {metrics_path}")
    print(f"Saved scaled training objectives to {objective_scales_path}")
    print(f"Saved ODE model checkpoint to {ode_model_path}")
    print(f"Saved ODE training loss to {ode_history_path}")
    print(f"Saved SDE model checkpoint to {sde_model_path}")
    print(f"Saved SDE training loss to {sde_history_path}")
    print(f"Saved classical logistic parameters to {classical_parameters_path}")
    print(f"Saved classical logistic training loss to {classical_history_path}")
    if plot:
        print(f"Saved comparison plot to {plot_path}")
        print(f"Saved training objective plot to {objective_plot_path}")


def main() -> None:
    """Run deterministic and stochastic neural dynamics comparison."""

    args = parse_args()
    if args.ode_epochs < 1:
        raise ValueError("ode_epochs must be at least 1.")
    if args.sde_epochs < 1:
        raise ValueError("sde_epochs must be at least 1.")
    if args.classical_epochs < 1:
        raise ValueError("classical_epochs must be at least 1.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    data_path = resolve_data_path(args.data_path)
    frame = read_trial_csv(data_path)
    frame, trial_columns = validate_frame(frame)
    data = build_training_data(frame, trial_columns)

    print("Training classical logistic growth baseline...")
    classical_fit = train_classical_logistic_model(
        data=data,
        learning_rate=args.classical_learning_rate,
        epochs=args.classical_epochs,
        device=device,
    )

    print("Training deterministic Neural ODE...")
    ode_model, ode_prediction, ode_history = train_model(
        data=data,
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        learning_rate=args.ode_learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.ode_epochs,
        method=args.ode_method,
        device=device,
    )

    print("Training stochastic Neural SDE...")
    sde_model, sde_history = train_sde_model(
        data=data,
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        learning_rate=args.sde_learning_rate,
        weight_decay=args.weight_decay,
        min_diffusion=args.min_diffusion,
        epochs=args.sde_epochs,
        device=device,
    )
    sde_prediction = simulate_sde_paths(
        model=sde_model,
        data=data,
        paths=args.sde_paths,
        interval_level=args.interval_level,
        seed=args.seed + 1,
        device=device,
    )

    comparison_frame = build_comparison_frame(
        data=data,
        ode_prediction=ode_prediction,
        sde_prediction=sde_prediction,
        classical_fit=classical_fit,
    )
    metrics_frame = build_metrics_frame(
        data=data,
        comparison_frame=comparison_frame,
        ode_model=ode_model,
        ode_history=ode_history,
        sde_model=sde_model,
        sde_history=sde_history,
        classical_fit=classical_fit,
    )
    save_outputs(
        output_dir=args.output_dir,
        data=data,
        comparison_frame=comparison_frame,
        metrics_frame=metrics_frame,
        ode_model=ode_model,
        ode_history=ode_history,
        sde_model=sde_model,
        sde_history=sde_history,
        classical_fit=classical_fit,
        interval_level=args.interval_level,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
