from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from kefir_models import logistic_pinn_compare
from kefir_models.ode_fit import build_training_data
from kefir_models.sde_compare import train_classical_logistic_model


def test_parse_args_help(capsys):
    """Test that the logistic PINN parser exposes help text."""

    try:
        logistic_pinn_compare.parse_args(["--help"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "deterministic and stochastic logistic PINN" in captured.out


def toy_data():
    frame = pd.DataFrame(
        {"time": [5.0, 17.0, 29.0], "trial_1": [10.0, 17.0, 23.0], "trial_2": [11.0, 18.0, 24.0]}
    )
    return logistic_pinn_compare.build_logistic_data(
        frame, ["trial_1", "trial_2"], torch.device("cpu")
    )


def test_capacity_bound_matches_classical_domain_not_largest_response():
    data = toy_data()
    assert data.minimum_capacity == pytest.approx(11.001)
    model, parameters = logistic_pinn_compare.make_model_and_parameters(data, 4, 1, False, 0.02)
    assert float(parameters.minimum_capacity) == pytest.approx(11.001)
    # A capacity below a noisy final observation must remain admissible.
    parameters.raw_capacity_margin.data.fill_(1.0)
    assert float(parameters.carrying_capacity) < 24.0
    assert bool((model(data.normalized_time) > 0).all())


def test_physics_residual_obeys_normalized_time_chain_rule():
    data = toy_data()
    parameters = logistic_pinn_compare.LogisticParameterization(11.001, 30.0, 0.04)

    class ExactLogistic(nn.Module):
        def forward(self, tau):
            k = parameters.carrying_capacity
            return k / (
                1 + (k / 10.0 - 1) * torch.exp(-parameters.growth_rate * data.time_scale * tau)
            )

    tau = torch.linspace(0, 1, 15).reshape(-1, 1).requires_grad_(True)
    residual = logistic_pinn_compare.physics_residual_loss(ExactLogistic(), parameters, tau, data)
    assert float(residual) < 1e-9
    wrong_units = replace(data, time_scale=1.0)
    assert (
        float(
            logistic_pinn_compare.physics_residual_loss(
                ExactLogistic(), parameters, tau, wrong_units
            )
        )
        > 0.1
    )


def simulation(times, step, sigma=0.0, paths=1):
    return logistic_pinn_compare.simulate_logistic_sde(
        np.asarray(times),
        np.array([14.0, 14.8]),
        0.036806,
        52.2809,
        sigma,
        paths,
        0.9,
        8,
        torch.device("cpu"),
        max_step=step,
    )


def test_logistic_deterministic_limit_converges_to_exact_solution():
    times = np.arange(0.0, 169.0, 12.0)
    exact = 52.2809 / (
        1 + (52.2809 / np.array([14.0, 14.8]) - 1)[None, :] * np.exp(-0.036806 * times[:, None])
    )
    coarse = simulation(times, 12.0)
    fine = simulation(times, 0.05)
    assert np.max(np.abs(fine.mean_by_trial - exact)) < 0.005
    assert np.max(np.abs(fine.mean_by_trial - exact)) < (
        np.max(np.abs(coarse.mean_by_trial - exact)) / 100
    )
    assert fine.clipped_values == 0


def test_plot_resolution_does_not_change_stochastic_values_at_shared_times():
    coarse = simulation([0.0, 12.0, 24.0], 0.05, sigma=0.08, paths=128)
    dense = simulation([0.0, 3.0, 12.0, 17.0, 24.0], 0.05, sigma=0.08, paths=128)
    for field in ("mean_by_trial", "lower_by_trial", "upper_by_trial"):
        np.testing.assert_array_equal(getattr(coarse, field), getattr(dense, field)[[0, 2, 4]])


def test_classical_solver_recovers_synthetic_logistic_parameters():
    times = np.linspace(0.0, 100.0, 15)
    initial = np.array([8.0, 10.0])
    exact = 40.0 / (1 + (40.0 / initial - 1)[None, :] * np.exp(-0.05 * times[:, None]))
    frame = pd.DataFrame({"time": times, "trial_1": exact[:, 0], "trial_2": exact[:, 1]})
    fit = train_classical_logistic_model(
        build_training_data(frame, ["trial_1", "trial_2"]), 0.05, 1800, torch.device("cpu")
    )
    assert fit.growth_rate / 100 == pytest.approx(0.05, rel=0.002)
    assert fit.carrying_capacity == pytest.approx(40.0, rel=0.002)


def test_pinn_metrics_include_complete_report_criteria():
    data = toy_data()
    model, parameters = logistic_pinn_compare.make_model_and_parameters(data, 4, 1, True, 0.02)
    fit = logistic_pinn_compare.LogisticPINNFit(model, parameters, pd.DataFrame(), 0.0)
    frame = pd.DataFrame(
        {"deterministic_pinn": [10.5, 17.5, 23.5], "stochastic_pinn_drift": [10.5, 17.5, 23.5]}
    )
    for column in data.trial_columns:
        values = data.raw_frame[column].to_numpy()
        frame[f"{column}_sde_mean"] = values + 0.5
        frame[f"{column}_sde_lower"] = values - 1
        frame[f"{column}_sde_upper"] = values + 1
    metrics = logistic_pinn_compare.build_metrics_frame(data, frame, fit, fit)
    assert np.isfinite(metrics[["rmse", "aic", "bic", "residual_sigma"]]).all().all()
    sde = metrics.iloc[-1]
    assert sde.residual_sigma == pytest.approx(0.5)
    assert sde.observation_count == 6
    expected_deviance = 6 * (np.log(2 * np.pi) + 1 + np.log(0.25))
    assert sde.aic == pytest.approx(expected_deviance + 2 * sde.likelihood_parameter_count)
