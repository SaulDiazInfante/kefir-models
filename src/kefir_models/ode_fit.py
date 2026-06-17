"""Fit water kefir trial data with a Neural Ordinary Differential Equation.

The script trains a non-autonomous Neural ODE

    dy / dt = f_theta(t, y)

against all trial columns in ``waterKefirTrialsReference.csv`` using one
population trajectory. The single initial value is the mean of the first
observed replicate values, and the same fitted curve is compared against every
trial observation. The current misspelled file name
``waterKefirTrialsReferece.csv`` is accepted as a fallback. The script writes
the fitted population curve, a trained model checkpoint, and a diagnostic plot
to the selected output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torchdiffeq import odeint

from kefir_models.plot_ode_fit import save_plot

DEFAULT_DATA_PATH = Path("data/raw/waterKefirTrialsReference.csv")
FALLBACK_DATA_PATH = Path("data/raw/waterKefirTrialsReferece.csv")


@dataclass(frozen=True)
class Scaling:
    """Affine scaling constants used during training."""

    time_min: float
    time_scale: float
    value_mean: float
    value_std: float


@dataclass(frozen=True)
class TrainingData:
    """Tensor representation of the observed trial trajectories."""

    time: Tensor
    values: Tensor
    mask: Tensor
    scaling: Scaling
    trial_columns: list[str]
    raw_frame: pd.DataFrame


class NeuralODEFunction(nn.Module):
    """Neural right-hand side for dy / dt = f_theta(t, y)."""

    def __init__(self, hidden_units: int, hidden_layers: int) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")

        layers: list[nn.Module] = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
        layers.append(nn.Linear(hidden_units, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        """Return dy/dt for a batch of one-dimensional states."""

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        time_column = torch.ones_like(y[..., :1]) * t
        features = torch.cat((time_column, y), dim=-1)
        return self.network(features)


def load_args_from_json(json_path: Path) -> dict:
    """Load arguments from a JSON configuration file."""

    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Fit water kefir trial data with a Neural ODE.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON configuration file. Overrides other arguments.",
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
        default=Path("neural_ode_outputs"),
        help="Directory where outputs are written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Adam optimizer learning rate.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=32,
        help="Width of each hidden layer in the ODE function.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=2,
        help="Number of hidden layers in the ODE function.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization used by Adam.",
    )
    parser.add_argument(
        "--method",
        choices=("dopri5", "rk4"),
        default="dopri5",
        help="ODE solver used by torchdiffeq.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible training.",
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
        help="Whether to save a diagnostic plot.",
    )
    temp_args, _ = parser.parse_known_args(args)
    if temp_args.config is not None:
        config_data = load_args_from_json(temp_args.config)
        if "data_path" in config_data:
            config_data["data_path"] = Path(config_data["data_path"])
        if "output_dir" in config_data:
            config_data["output_dir"] = Path(config_data["output_dir"])
        parser.set_defaults(**config_data)

    return parser.parse_args(args)


def read_trial_csv(path: Path) -> pd.DataFrame:
    """Read a trial CSV, trimming accidental trailing fields in bad rows."""

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Data file is empty: {path}") from exc

        header = [column.strip() for column in header]
        expected_columns = len(header)
        for row in reader:
            if not row or all(not value.strip() for value in row):
                continue
            trimmed_row = row[:expected_columns]
            padded_row = trimmed_row + [""] * (expected_columns - len(trimmed_row))
            rows.append(padded_row)

    frame = pd.DataFrame(rows, columns=header)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def resolve_data_path(path: Path) -> Path:
    """Resolve the requested data path, including the known misspelled file."""

    if path.exists():
        return path
    if path == DEFAULT_DATA_PATH and FALLBACK_DATA_PATH.exists():
        print(
            "Using fallback data file "
            f"{FALLBACK_DATA_PATH} because {DEFAULT_DATA_PATH} was not found.",
        )
        return FALLBACK_DATA_PATH
    return path


def validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and sort the observed trial data."""

    if "time" not in frame.columns:
        raise ValueError("The CSV must contain a 'time' column.")

    trial_columns = [column for column in frame.columns if column != "time"]
    if not trial_columns:
        raise ValueError("The CSV must contain at least one trial column.")

    frame = (
        frame.dropna(subset=["time"])
        .sort_values("time")
        .reset_index(
            drop=True,
        )
    )
    if frame.empty:
        raise ValueError("No valid time points were found.")

    if frame["time"].duplicated().any():
        raise ValueError("The time column contains duplicate values.")

    values = frame[trial_columns].to_numpy(dtype=np.float64)
    if np.isnan(values).all():
        raise ValueError("No numeric trial values were found.")

    first_row = values[0]
    if np.isnan(first_row).any():
        raise ValueError("The first row must have all trial initial values.")

    return frame, trial_columns


def select_device(device_name: str) -> torch.device:
    """Return the training device requested by the user."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_name)


def build_training_data(frame: pd.DataFrame, trial_columns: list[str]) -> TrainingData:
    """Convert a validated data frame into normalized training tensors."""

    time_values = frame["time"].to_numpy(dtype=np.float32)
    trial_values = frame[trial_columns].to_numpy(dtype=np.float32)

    observed_values = trial_values[~np.isnan(trial_values)]
    value_mean = float(observed_values.mean())
    value_std = float(observed_values.std(ddof=0))
    if value_std == 0.0:
        value_std = 1.0

    time_min = float(time_values.min())
    time_scale = float(time_values.max() - time_min)
    if time_scale == 0.0:
        time_scale = 1.0

    normalized_time = (time_values - time_min) / time_scale
    normalized_values = (trial_values - value_mean) / value_std

    mask_array = ~np.isnan(normalized_values)
    normalized_values = np.nan_to_num(normalized_values, nan=0.0)

    values = torch.tensor(normalized_values[..., None], dtype=torch.float32)
    mask = torch.tensor(mask_array[..., None], dtype=torch.bool)

    return TrainingData(
        time=torch.tensor(normalized_time, dtype=torch.float32),
        values=values,
        mask=mask,
        scaling=Scaling(
            time_min=time_min,
            time_scale=time_scale,
            value_mean=value_mean,
            value_std=value_std,
        ),
        trial_columns=trial_columns,
        raw_frame=frame,
    )


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over observed entries only."""

    residual = prediction[mask] - target[mask]
    return torch.mean(residual.pow(2))


def population_initial_state(target: Tensor) -> Tensor:
    """Return the single normalized initial state for the population curve."""

    return target[0].mean(dim=0, keepdim=True)


def population_masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Compare one fitted trajectory against all observed trial values."""

    return masked_mse(prediction.expand_as(target), target, mask)


def solve_trajectory(
    ode_function: NeuralODEFunction,
    initial_state: Tensor,
    time_points: Tensor,
    method: str,
) -> Tensor:
    """Integrate the Neural ODE at requested time points."""

    if method == "rk4":
        return odeint(
            ode_function,
            initial_state,
            time_points,
            method=method,
            options={"step_size": 0.01},
        )

    return odeint(
        ode_function,
        initial_state,
        time_points,
        method=method,
        rtol=1e-5,
        atol=1e-6,
    )


def train_model(
    data: TrainingData,
    hidden_units: int,
    hidden_layers: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    method: str,
    device: torch.device,
) -> tuple[NeuralODEFunction, Tensor, list[float]]:
    """Train a Neural ODE and return one fitted normalized population curve."""

    model = NeuralODEFunction(
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    time_points = data.time.to(device)
    target = data.values.to(device)
    mask = data.mask.to(device)
    initial_state = population_initial_state(target)
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = solve_trajectory(
            model,
            initial_state,
            time_points,
            method=method,
        )
        loss = population_masked_mse(prediction, target, mask)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if epoch == 1 or epoch % 250 == 0 or epoch == epochs:
            print(f"epoch={epoch:05d} loss={loss_value:.6f}")

    model.eval()
    with torch.no_grad():
        prediction = solve_trajectory(
            model,
            initial_state,
            time_points,
            method=method,
        )

    return model, prediction.detach().cpu(), history


def inverse_transform_values(values: np.ndarray, scaling: Scaling) -> np.ndarray:
    """Map normalized values back to the original data scale."""

    return values * scaling.value_std + scaling.value_mean


def build_output_frame(data: TrainingData, prediction: Tensor) -> pd.DataFrame:
    """Build a wide output frame with observations and one fitted curve."""

    fitted_values = inverse_transform_values(
        prediction.numpy().reshape(-1),
        data.scaling,
    )

    output = pd.DataFrame({"time": data.raw_frame["time"].to_numpy()})
    for column in data.trial_columns:
        output[f"{column}_observed"] = data.raw_frame[column].to_numpy()

    output["observed_mean"] = data.raw_frame[data.trial_columns].mean(axis=1)
    output["fitted"] = fitted_values
    return output


def compute_rmse(output_frame: pd.DataFrame, trial_columns: Iterable[str]) -> float:
    """Compute RMSE on the original data scale."""

    residuals = []
    for column in trial_columns:
        observed = output_frame[f"{column}_observed"].to_numpy()
        fitted = output_frame["fitted"].to_numpy()
        mask = ~np.isnan(observed)
        residuals.append(observed[mask] - fitted[mask])

    all_residuals = np.concatenate(residuals)
    return float(np.sqrt(np.mean(all_residuals**2)))


def save_artifacts(
    output_dir: Path,
    model: NeuralODEFunction,
    data: TrainingData,
    prediction: Tensor,
    history: list[float],
    plot: bool,
) -> None:
    """Persist fitted values, model parameters, loss history, and plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_frame = build_output_frame(data, prediction)
    rmse = compute_rmse(output_frame, data.trial_columns)

    fitted_path = output_dir / "water_kefir_neural_ode_fit.csv"
    model_path = output_dir / "water_kefir_neural_ode_model.pt"
    history_path = output_dir / "training_loss.csv"
    plot_path = output_dir / "water_kefir_neural_ode_fit.png"

    history_frame = pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "loss": history})
    output_frame.to_csv(fitted_path, index=False)
    history_frame.to_csv(history_path, index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "trial_columns": data.trial_columns,
            "scaling": data.scaling.__dict__,
            "initial_state": population_initial_state(data.values).numpy().tolist(),
            "curve_type": "single_population_curve",
        },
        model_path,
    )

    if plot:
        save_plot(output_frame, data.trial_columns, plot_path, history_frame)

    print(f"RMSE={rmse:.6f}")
    print(f"Saved fitted values to {fitted_path}")
    print(f"Saved model checkpoint to {model_path}")
    print(f"Saved training loss to {history_path}")
    if plot:
        print(f"Saved plot to {plot_path}")


def main() -> None:
    """Run Neural ODE fitting from the command line."""

    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    data_path = resolve_data_path(args.data_path)
    frame = read_trial_csv(data_path)
    frame, trial_columns = validate_frame(frame)
    data = build_training_data(frame, trial_columns)

    model, prediction, history = train_model(
        data=data,
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        method=args.method,
        device=device,
    )
    save_artifacts(
        output_dir=args.output_dir,
        model=model,
        data=data,
        prediction=prediction,
        history=history,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
