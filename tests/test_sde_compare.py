import math

import pandas as pd
import pytest
import torch

from kefir_models import sde_compare
from kefir_models.ode_fit import Scaling, TrainingData


def test_training_objective_scales_frame_converts_histories():
    """Objective histories are converted between normalized and original scales."""

    data = TrainingData(
        time=torch.tensor([0.0, 1.0]),
        values=torch.zeros((2, 1, 1)),
        mask=torch.ones((2, 1, 1), dtype=torch.bool),
        scaling=Scaling(
            time_min=0.0,
            time_scale=1.0,
            value_mean=10.0,
            value_std=2.0,
        ),
        trial_columns=["trial_1"],
        raw_frame=pd.DataFrame({"time": [0.0, 1.0], "trial_1": [10.0, 12.0]}),
    )

    frame = sde_compare.build_training_objective_scales_frame(
        data=data,
        classical_history=[8.0],
        ode_history=[3.0],
        sde_history=[1.0],
    )

    classical = frame[frame["model"] == "Classical logistic ODE"].iloc[0]
    assert classical["normalized_value"] == pytest.approx(2.0)
    assert classical["original_scale_value"] == pytest.approx(8.0)

    ode = frame[frame["model"] == "Neural ODE"].iloc[0]
    assert ode["normalized_value"] == pytest.approx(3.0)
    assert ode["original_scale_value"] == pytest.approx(12.0)

    sde = frame[frame["model"] == "Neural SDE"].iloc[0]
    assert sde["normalized_value"] == pytest.approx(1.0)
    assert sde["original_scale_value"] == pytest.approx(1.0 + math.log(2.0))
