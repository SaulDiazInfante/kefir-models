"""Unit tests for deterministic helpers in :mod:`kefir_models.ode_fit`."""

import numpy as np
import pandas as pd
import pytest
import torch

from kefir_models import ode_fit
from kefir_models.ode_fit import NeuralODEFunction, Scaling


def test_validate_frame_sorts_and_lists_trial_columns():
    frame = pd.DataFrame({"time": [1.0, 0.0], "trial_1": [2.0, 1.0], "trial_2": [3.0, 1.5]})
    sorted_frame, trial_columns = ode_fit.validate_frame(frame)
    assert trial_columns == ["trial_1", "trial_2"]
    assert list(sorted_frame["time"]) == [0.0, 1.0]


def test_validate_frame_requires_time_column():
    with pytest.raises(ValueError, match="time"):
        ode_fit.validate_frame(pd.DataFrame({"trial_1": [1.0]}))


def test_validate_frame_rejects_duplicate_times():
    frame = pd.DataFrame({"time": [0.0, 0.0], "trial_1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        ode_fit.validate_frame(frame)


def test_validate_frame_requires_complete_first_row():
    frame = pd.DataFrame({"time": [0.0, 1.0], "trial_1": [np.nan, 2.0]})
    with pytest.raises(ValueError, match="first row"):
        ode_fit.validate_frame(frame)


def test_build_training_data_normalizes_time_and_values():
    frame, trial_columns = ode_fit.validate_frame(
        pd.DataFrame({"time": [0.0, 1.0, 2.0], "trial_1": [1.0, 2.0, 3.0]})
    )
    data = ode_fit.build_training_data(frame, trial_columns)

    assert isinstance(data.scaling, Scaling)
    assert data.scaling.time_min == 0.0
    assert data.scaling.time_scale == 2.0
    assert data.scaling.value_mean == pytest.approx(2.0)
    assert data.values.shape == (3, 1, 1)
    assert bool(data.mask.all())
    assert float(data.time[0]) == 0.0
    assert float(data.time[-1]) == pytest.approx(1.0)


def test_build_training_data_masks_missing_interior_values():
    frame, trial_columns = ode_fit.validate_frame(
        pd.DataFrame({"time": [0.0, 1.0, 2.0], "trial_1": [1.0, np.nan, 3.0]})
    )
    data = ode_fit.build_training_data(frame, trial_columns)
    assert bool(data.mask[1, 0, 0]) is False
    assert float(data.values[1, 0, 0]) == 0.0


def test_masked_mse_uses_observed_entries_only():
    prediction = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[1.0], [5.0]])
    mask = torch.tensor([[True], [False]])
    assert float(ode_fit.masked_mse(prediction, target, mask)) == 0.0


def test_population_initial_state_averages_first_row():
    target = torch.tensor([[[1.0], [3.0]]])  # shape (1, 2, 1)
    initial = ode_fit.population_initial_state(target)
    assert initial.shape == (1, 1)
    assert float(initial) == pytest.approx(2.0)


def test_inverse_transform_values_round_trips_scaling():
    scaling = Scaling(time_min=0.0, time_scale=1.0, value_mean=10.0, value_std=2.0)
    out = ode_fit.inverse_transform_values(np.array([0.0, 1.0, -1.0]), scaling)
    assert list(out) == [10.0, 12.0, 8.0]


def test_compute_rmse_skips_missing_observations():
    frame = pd.DataFrame({"trial_1_observed": [1.0, 2.0, np.nan], "fitted": [1.0, 4.0, 9.0]})
    assert ode_fit.compute_rmse(frame, ["trial_1"]) == pytest.approx(2.0**0.5)


def test_resolve_data_path_returns_existing(tmp_path):
    path = tmp_path / "trials.csv"
    path.write_text("time,trial_1\n0,1\n")
    assert ode_fit.resolve_data_path(path) == path


def test_read_trial_csv_trims_extra_fields_and_parses(tmp_path):
    path = tmp_path / "trials.csv"
    path.write_text("time,trial_1\n0,1\n1,2,99\n\n")
    frame = ode_fit.read_trial_csv(path)
    assert list(frame.columns) == ["time", "trial_1"]
    assert frame.shape == (2, 2)
    assert frame["trial_1"].tolist() == [1.0, 2.0]


def test_read_trial_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ode_fit.read_trial_csv(tmp_path / "missing.csv")


def test_neural_ode_function_forward_shape():
    torch.manual_seed(0)
    function = NeuralODEFunction(hidden_units=4, hidden_layers=1)
    out = function(torch.tensor(0.5), torch.zeros((3, 1)))
    assert out.shape == (3, 1)


def test_neural_ode_function_requires_one_layer():
    with pytest.raises(ValueError):
        NeuralODEFunction(hidden_units=4, hidden_layers=0)


def test_select_device_cpu():
    assert ode_fit.select_device("cpu").type == "cpu"
