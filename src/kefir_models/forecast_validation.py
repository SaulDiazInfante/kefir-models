"""Time-ordered data preparation for the REV-07 controlled experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from kefir_models.ode_fit import TrainingData, build_training_data


def training_and_evaluation(
    frame: pd.DataFrame, columns: list[str], cutoff: float
) -> tuple[TrainingData, TrainingData]:
    """Fit scaling on early times, then apply it unchanged to all observations."""

    train_frame = frame.loc[frame.time <= cutoff].copy().reset_index(drop=True)
    if len(train_frame) < 2 or not np.isclose(train_frame.time.iloc[-1], cutoff):
        raise ValueError("cutoff must be an observed time with at least two training rows.")
    train = build_training_data(train_frame, columns)
    scaling = train.scaling
    values = frame[columns].to_numpy(dtype=np.float32)
    mask = ~np.isnan(values)
    standardized = (values - scaling.value_mean) / scaling.value_std
    evaluation = TrainingData(
        time=torch.tensor(
            (frame.time.to_numpy(dtype=np.float32) - scaling.time_min) / scaling.time_scale
        ),
        values=torch.tensor(np.nan_to_num(standardized, nan=0.0)[..., None]),
        mask=torch.tensor(mask[..., None]),
        scaling=scaling,
        trial_columns=columns,
        raw_frame=frame.copy(),
    )
    return train, evaluation


def forecast_data(evaluation: TrainingData, cutoff: float) -> TrainingData:
    """Condition a free forecast on the last training observation, retaining time origin."""

    indices = np.flatnonzero(evaluation.raw_frame.time.to_numpy() >= cutoff)
    return TrainingData(
        time=evaluation.time[indices],
        values=evaluation.values[indices],
        mask=evaluation.mask[indices],
        scaling=evaluation.scaling,
        trial_columns=evaluation.trial_columns,
        raw_frame=evaluation.raw_frame.iloc[indices].reset_index(drop=True),
    )
