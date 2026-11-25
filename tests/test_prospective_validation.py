import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from kefir_models.bayesian_observation import log_median, reference_prior
from kefir_models.ode_fit import read_trial_csv, validate_frame
from kefir_models.prospective_validation import (
    aggregate_units,
    audit_plan,
    freeze_protocol,
    gaussian_forecast,
    score_mixture,
    sha256,
    verify_protocol,
)
from kefir_models.structural_discrepancy import StructuralProblem

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("sharing", ["shared", "independent"])
@pytest.mark.parametrize("kernel", ["ou", "matern32"])
def test_outcome_free_forecast_retains_exact_joint_density(sharing, kernel):
    z = reference_prior("ou").mean
    times = np.array([0.0, 1.0])
    values = np.repeat(np.exp(log_median(times, z, "ou"))[:, None], 3, axis=1)
    problem = StructuralProblem(times, values, kernel, sharing)
    future = np.array([2.0, 3.0])
    means, covs = gaussian_forecast(problem, z[None, None, :], future, 1)
    observed = np.exp(means[0] + np.linspace(-0.01, 0.02, 6))
    pointwise, joint = score_mixture(means, covs, observed)
    expected = -problem.joint_log_prediction(z, future, observed.reshape(2, 3)) / 6
    assert joint == pytest.approx(expected, abs=1e-11)
    np.testing.assert_allclose(pointwise["point"], np.exp(means[0]))
    with pytest.raises(ValueError, match="follow the training"):
        gaussian_forecast(problem, z[None, None, :], np.array([1.0, 2.0]), 1)


def test_mixture_joint_score_includes_response_jacobian_and_parameter_integration():
    means = np.array([[1.0, 2.0], [0.8, 2.3]])
    covariance = np.array([[[0.1, 0.03], [0.03, 0.2]], [[0.2, -0.01], [-0.01, 0.12]]])
    observed = np.exp([0.9, 2.1])
    _, result = score_mixture(means, covariance, observed)
    densities = [
        multivariate_normal.logpdf(np.log(observed), mean=m, cov=c) - np.log(observed).sum()
        for m, c in zip(means, covariance)
    ]
    assert result == pytest.approx(-(logsumexp(densities) - np.log(2)) / 2)
    with pytest.raises(ValueError, match="positive response"):
        score_mixture(means, covariance, np.array([0.0, 2.0]))


@pytest.mark.parametrize("sharing", ["independent", "shared"])
def test_measurement_sd_tail_has_polynomial_likelihood_decay(sharing):
    z = reference_prior("ou").mean
    time = np.arange(4.0)
    values = np.repeat(np.exp(log_median(time, z, "ou"))[:, None], 3, axis=1)
    problem = StructuralProblem(time, values, "ou", sharing)
    densities, integrands = [], []
    for level in [5.0, 6.0]:
        point = z.copy()
        point[-1] = level
        densities.append(problem.log_likelihood(point))
        mean, latent, noise = problem.conditional_log_distribution(point, np.array([4.0]))
        integrands.append(problem.log_posterior(point) + mean[0, 0] + 0.5 * (latent[0, 0] + noise))
    # The finite check verifies L(b) ~ constant*b^(-N). Divergence of
    # the response mean under the lognormal SD prior is established analytically.
    assert densities[1] - densities[0] == pytest.approx(-values.size, abs=1e-6)
    assert integrands[1] > integrands[0] > 1000.0


def score_rows():
    rows = []
    for model in ["a", "b"]:
        for unit, size, error in [("unit_a", 1, 100.0), ("unit_b", 9, 0.0)]:
            for origin in range(size):
                rows.append(
                    {
                        "model": model,
                        "unit_id": unit,
                        "origin": origin,
                        "trial": "trial_1",
                        "horizon": 12.0,
                        "squared_error": error,
                        "mean_bias": 0.0,
                        "coverage": 0.9,
                        "mean_width": 1.0,
                        "interval_score": 1.0,
                        "marginal_predictive_nll": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregation_weights_units_equally_not_observation_counts():
    unit, summary = aggregate_units(score_rows())
    assert len(unit) == 4
    np.testing.assert_allclose(summary.rmse, np.sqrt(50.0))
    assert (summary.independent_units == 2).all()


def test_unpaired_or_duplicate_targets_cannot_enter_comparison():
    rows = score_rows()
    with pytest.raises(ValueError, match="identical targets"):
        aggregate_units(rows.iloc[1:])
    with pytest.raises(ValueError, match="once"):
        aggregate_units(pd.concat([rows, rows.iloc[:1]]))
    rows.loc[rows.model == "b", "horizon"] = 24.0
    with pytest.raises(ValueError, match="same candidate"):
        aggregate_units(rows)


@pytest.fixture
def draft(tmp_path):
    plan = json.loads((REPO / "configs/prospective_validation.json").read_text())
    metadata = json.loads((REPO / "data/metadata/reference.json").read_text())
    metadata["path"] = "reference.csv"
    (tmp_path / "reference.csv").write_bytes(
        (REPO / "data/raw/waterKefirTrialsReferece.csv").read_bytes()
    )
    plan["reference_metadata"] = "metadata.json"
    plan["origins"] = [12.0]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return path, plan, metadata


def confirm_synthetic_fixture(draft):
    path, plan, metadata = draft
    for key in [
        "time_unit",
        "response_name",
        "response_unit",
        "experimental_unit_definition",
        "shared_conditions",
        "confirmation_source",
    ]:
        metadata[key] = "synthetic test fixture"
    metadata["trial_to_experimental_unit"] = {"trial_1": "synthetic batch"}
    plan["independence_confirmed_by"] = "synthetic test fixture"
    plan["validation_units"] = [
        {
            "unit_id": "new_unit",
            "provenance": "synthetic",
            "independence_basis": "synthetic",
            "outcomes_uninspected_declaration": "synthetic",
            "time_unit": metadata["time_unit"],
            "response_unit": metadata["response_unit"],
        }
    ]
    path.write_text(json.dumps(plan))
    (path.parent / "metadata.json").write_text(json.dumps(metadata))
    return path, plan


def test_unknown_metadata_cannot_be_frozen(draft):
    path, _, _ = draft
    report = audit_plan(path, path.parent)
    assert not report["protocol_ready_to_freeze"]
    with pytest.raises(ValueError, match="incomplete"):
        freeze_protocol(path, path.parent / "frozen", path.parent)
    assert not (path.parent / "frozen").exists()


def test_protocol_can_be_frozen_before_future_prefixes_exist(draft, monkeypatch):
    path, _ = confirm_synthetic_fixture(draft)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Freeze read observations"))
    freeze_protocol(path, path.parent / "frozen", path.parent)
    assert (
        verify_protocol(path.parent / "frozen")["status"]
        == "protocol_frozen_validation_not_performed"
    )
    with pytest.raises(FileExistsError):
        freeze_protocol(path, path.parent / "frozen", path.parent)
    frozen = path.parent / "frozen/plan.json"
    frozen.write_text(frozen.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        verify_protocol(path.parent / "frozen")


def test_prefix_audit_rejects_future_rows_and_renamed_development_data(draft):
    path, plan = confirm_synthetic_fixture(draft)
    plan["validation_units"][0]["prefixes"] = {"12": "prefix.csv"}
    path.write_text(json.dumps(plan))
    pd.DataFrame({"time": [0.0, 12.0, 24.0], "trial_1": [10.0, 12.0, 15.0]}).to_csv(
        path.parent / "prefix.csv", index=False
    )
    with pytest.raises(ValueError, match="no future rows"):
        audit_plan(path, path.parent)
    development, columns = validate_frame(read_trial_csv(path.parent / "reference.csv"))
    development.loc[development.time <= 12, ["time", *columns]].to_csv(
        path.parent / "prefix.csv", index=False
    )
    with pytest.raises(ValueError, match="development-data prefix"):
        audit_plan(path, path.parent)
    pd.DataFrame({"time": [0.0, 12.0], "trial_1": [10.0, 12.0]}).to_csv(
        path.parent / "prefix.csv", index=False
    )
    report = audit_plan(path, path.parent)
    assert report["status"] == "ready_for_prefix_forecasts"
    assert report["validated_prefixes"][0]["sha256"] == sha256(path.parent / "prefix.csv")
