import pandas as pd
import pytest
import torch

from kefir_models.forecast_validation import forecast_data, training_and_evaluation


def test_future_values_do_not_affect_training_scaling():
    frame = pd.DataFrame({"time": [5.0, 17.0, 29.0, 41.0], "trial_1": [10.0, 20.0, 30.0, 40.0]})
    train, evaluation = training_and_evaluation(frame, ["trial_1"], 29.0)
    changed = frame.copy()
    changed.loc[3, "trial_1"] = 1e9
    other_train, _ = training_and_evaluation(changed, ["trial_1"], 29.0)
    assert train.scaling == other_train.scaling
    assert evaluation.scaling == train.scaling
    torch.testing.assert_close(train.values, other_train.values)
    forecast = forecast_data(evaluation, 29.0)
    assert forecast.time[0] == 1.0
    assert forecast.time[-1] == 1.5
    assert list(forecast.raw_frame.time) == [29.0, 41.0]


def test_cutoff_must_be_an_observed_training_time():
    frame = pd.DataFrame({"time": [0.0, 12.0, 24.0], "trial_1": [10.0, 20.0, 30.0]})
    with pytest.raises(ValueError):
        training_and_evaluation(frame, ["trial_1"], 17.0)
