import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from kefir_models.bayesian_observation import ObservationProblem, log_median, reference_prior
from kefir_models.structural_discrepancy import StructuralProblem


def predictor():
    path = Path(__file__).resolve().parents[1] / "scripts/investigate_bayesian_observation.py"
    spec = importlib.util.spec_from_file_location("bayesian_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predictive_summary


def test_fixed_iid_predictive_matches_analytic_lognormal():
    z = reference_prior("iid").mean
    train_time = np.array([0.0, 1.0])
    train_values = np.exp(log_median(train_time, z, "iid"))[:, None]
    problem = ObservationProblem(train_time, train_values, "iid")
    times = np.array([2.0, 3.0])
    log_mean = log_median(times, z, "iid")
    observed = np.exp(log_mean)[:, None]
    score, output = predictor()(problem, z[None, None, :], times, observed, 1)
    np.testing.assert_allclose(output["mean"][:, 0], np.exp(log_mean + 0.5 * 0.03**2))
    np.testing.assert_allclose(output["lower"][:, 0], np.exp(log_mean + ndtri(0.05) * 0.03))
    np.testing.assert_allclose(output["upper"][:, 0], np.exp(log_mean + ndtri(0.95) * 0.03))
    expected = 0.5 * np.log(2 * np.pi * 0.03**2) + log_mean.mean()
    np.testing.assert_allclose(score["marginal_predictive_nll"], expected)
    assert score["parameter_log_variance"] == 0.0
    assert score["latent_log_variance"] == 0.0
    np.testing.assert_allclose(score["measurement_log_variance"], 0.03**2)


def test_single_future_ou_joint_and_marginal_scores_agree_for_fixed_parameters():
    z = reference_prior("ou").mean
    train_time = np.array([0.0, 1.0])
    values = np.repeat(np.exp(log_median(train_time, z, "ou"))[:, None], 3, axis=1)
    problem = ObservationProblem(train_time, values, "ou")
    times = np.array([2.0])
    observed = np.repeat(np.exp(log_median(times, z, "ou"))[:, None], 3, axis=1)
    score, _ = predictor()(problem, z[None, None, :], times, observed, 1)
    np.testing.assert_allclose(
        score["joint_predictive_nll_per_entry"], score["marginal_predictive_nll"]
    )


def test_shared_predictive_scoring_retains_cross_trial_dependence():
    z = reference_prior("ou").mean
    time = np.array([0.0, 1.0])
    values = np.repeat(np.exp(log_median(time, z, "ou"))[:, None], 3, axis=1)
    problem = StructuralProblem(time, values, "matern32", "shared")
    times = np.array([2.0, 3.0])
    observed = np.repeat(np.exp(log_median(times, z, "ou"))[:, None], 3, axis=1)
    score, prediction = predictor()(problem, z[None, None, :], times, observed, 1)
    np.testing.assert_allclose(
        score["joint_predictive_nll_per_entry"],
        -problem.joint_log_prediction(z, times, observed) / observed.size,
    )
    assert abs(score["joint_predictive_nll_per_entry"] - score["marginal_predictive_nll"]) > 0.01
    _, one = predictor()(problem, z[None, None, :], times[:1], observed[:1], 1)
    for key in ["mean", "lower", "upper"]:
        np.testing.assert_allclose(one[key], prediction[key][:1])


def test_structural_interval_score_penalizes_both_tails(monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    score = importlib.import_module("investigate_structural_discrepancy").interval_score
    np.testing.assert_allclose(
        score(np.array([0.0, 2.0, 5.0]), np.ones(3), np.full(3, 3.0)), [22.0, 2.0, 42.0]
    )
