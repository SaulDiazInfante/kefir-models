"""Unit tests for deterministic helpers in :mod:`kefir_models.sde_compare`."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from kefir_models import sde_compare
from kefir_models.ode_fit import Scaling, TrainingData
from kefir_models.sde_compare import NeuralSDEFunction, build_mlp


def _toy_training_data() -> TrainingData:
    return TrainingData(
        time=torch.tensor([0.0, 0.5, 1.0]),
        values=torch.tensor([[[0.0]], [[1.0]], [[2.0]]]),
        mask=torch.ones((3, 1, 1), dtype=torch.bool),
        scaling=Scaling(time_min=0.0, time_scale=1.0, value_mean=0.0, value_std=1.0),
        trial_columns=["trial_1"],
        raw_frame=pd.DataFrame({"time": [0.0, 0.5, 1.0], "trial_1": [0.0, 1.0, 2.0]}),
    )


def test_inverse_softplus_round_trips():
    raw = sde_compare.inverse_softplus(2.0)
    assert F.softplus(torch.tensor(raw)).item() == pytest.approx(2.0, abs=1e-5)


def test_inverse_softplus_rejects_non_positive():
    with pytest.raises(ValueError):
        sde_compare.inverse_softplus(0.0)


def test_logistic_growth_prediction_matches_closed_form():
    time = torch.tensor([0.0, 1.0, 2.0])
    initial = torch.tensor([1.0])
    prediction = sde_compare.logistic_growth_prediction(
        time, initial, torch.tensor(0.5), torch.tensor(10.0)
    )
    assert prediction.shape == (3, 1, 1)
    assert float(prediction[0, 0, 0]) == pytest.approx(1.0, rel=1e-4)
    assert float(prediction[2, 0, 0]) > float(prediction[0, 0, 0])


def test_compute_rmse_and_r_squared_perfect_fit():
    observed = np.array([1.0, 2.0, np.nan, 4.0])
    fitted = np.array([1.0, 2.0, 3.0, 4.0])
    assert sde_compare.compute_rmse(observed, fitted) == 0.0
    assert sde_compare.compute_r_squared(observed, fitted) == pytest.approx(1.0)


def test_observed_fitted_pairs_drops_missing():
    observed = np.array([1.0, np.nan, 3.0])
    fitted = np.array([1.0, 2.0, 3.0])
    obs, fit = sde_compare.observed_fitted_pairs(observed, fitted)
    assert list(obs) == [1.0, 3.0]
    assert list(fit) == [1.0, 3.0]


def test_gaussian_information_criteria_reasonable_values():
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    fitted = np.array([1.1, 1.9, 3.2, 3.8])
    sigma, aic, bic, count = sde_compare.gaussian_information_criteria(
        observed, fitted, parameter_count=2
    )
    assert count == 4
    assert sigma > 0.0
    # With n=4, log(n) < 2, so the BIC complexity penalty is below the AIC one.
    assert bic < aic


def test_gaussian_information_criteria_handles_no_observations():
    sigma, aic, bic, count = sde_compare.gaussian_information_criteria(
        np.array([np.nan]), np.array([1.0]), parameter_count=2
    )
    assert count == 0
    assert math.isnan(aic) and math.isnan(bic) and math.isnan(sigma)


def test_count_trainable_parameters_matches_manual_count():
    model = build_mlp(hidden_units=4, hidden_layers=1)
    # Linear(2,4): 8 + 4 ; Linear(4,1): 4 + 1 -> 17 trainable parameters.
    assert sde_compare.count_trainable_parameters(model) == 17


def test_compute_interval_coverage():
    observed = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 2.5, 2.0])
    upper = np.array([2.0, 3.0, 4.0])
    assert sde_compare.compute_interval_coverage(observed, lower, upper) == pytest.approx(2.0 / 3.0)


def test_population_curve_matrix_from_population_curve():
    matrix, curve = sde_compare.population_curve_matrix(np.array([1.0, 2.0, 3.0]), 3)
    assert matrix.shape == (3, 3)
    assert list(curve) == [1.0, 2.0, 3.0]


def test_population_curve_matrix_rejects_incompatible_shape():
    with pytest.raises(ValueError):
        sde_compare.population_curve_matrix(np.zeros((3, 2)), trial_count=3)


def test_build_transition_batch_shapes_and_deltas():
    batch = sde_compare.build_transition_batch(_toy_training_data(), torch.device("cpu"))
    assert batch.current.shape == (2, 1)
    assert batch.following.shape == (2, 1)
    assert torch.allclose(batch.delta_time, torch.tensor([[0.5], [0.5]]))


def test_transition_negative_log_likelihood_is_finite_scalar():
    torch.manual_seed(0)
    model = NeuralSDEFunction(hidden_units=4, hidden_layers=1, min_diffusion=1e-3)
    batch = sde_compare.build_transition_batch(_toy_training_data(), torch.device("cpu"))
    nll = sde_compare.transition_negative_log_likelihood(model, batch)
    assert nll.ndim == 0
    assert bool(torch.isfinite(nll))


def test_neural_sde_diffusion_respects_lower_bound():
    model = NeuralSDEFunction(hidden_units=4, hidden_layers=1, min_diffusion=0.01)
    diffusion = model.diffusion(torch.tensor(0.0), torch.zeros((5, 1)))
    assert bool(torch.all(diffusion >= 0.01))


@pytest.mark.parametrize(
    ("hidden_layers", "min_diffusion"),
    [(0, 1e-3), (1, 0.0)],
)
def test_neural_sde_rejects_invalid_arguments(hidden_layers, min_diffusion):
    with pytest.raises(ValueError):
        NeuralSDEFunction(
            hidden_units=4,
            hidden_layers=hidden_layers,
            min_diffusion=min_diffusion,
        )
