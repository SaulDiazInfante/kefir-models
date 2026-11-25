"""Outcome-free Gaussian-mixture forecasts and experimental-unit score summaries.

These utilities do not confirm experimental independence or supply new data.
The forecast constructor consumes a training problem and posterior samples;
future observed responses enter only the separate scoring operation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from kefir_models.posterior_prediction import gaussian_mixture_quantile


def gaussian_forecast(problem, chains, prediction_time, count=12000):
    """Return joint log-response mixture moments, without seeing target values.

    The time coordinates must use the same scale and origin as the problem.
    Entries are ordered by time, then trial. Each component retains all
    within-forecast covariance; components have equal posterior weights.
    """
    times = np.asarray(prediction_time, dtype=float)
    chains = np.asarray(chains, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Need finite prediction times.")
    if np.any(np.diff(times) <= 0) or times[0] <= problem.time[-1]:
        raise ValueError("Forecast times must increase and follow the training prefix.")
    if chains.ndim != 3 or chains.shape[-1] != 6 or not np.isfinite(chains).all():
        raise ValueError("Need chain-by-draw-by-six posterior samples.")
    flat = chains.reshape(-1, 6)
    if not len(flat) or count < 1:
        raise ValueError("Need at least one posterior component.")
    selected = flat[np.linspace(0, len(flat) - 1, min(count, len(flat)), dtype=int)]
    trials = problem.values.shape[1]
    sharing = problem.sharing
    groups = np.ones((trials, trials)) if sharing == "shared" else np.eye(trials)
    means, covariances = [], []
    for z in selected:
        mean, latent, noise = problem.conditional_log_distribution(z, times)
        means.append(mean.ravel())
        covariances.append(np.kron(latent, groups) + noise * np.eye(mean.size))
    return np.asarray(means), np.asarray(covariances)


def validate_mixture(means, covariances):
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if means.ndim != 2 or not all(means.shape):
        raise ValueError("Need component-by-target log means.")
    if covariances.shape != (means.shape[0], means.shape[1], means.shape[1]):
        raise ValueError("Need matching full component covariances.")
    if not np.isfinite(means).all() or not np.isfinite(covariances).all():
        raise ValueError("Mixture moments must be finite.")
    if not np.allclose(covariances, covariances.transpose(0, 2, 1), atol=1e-12, rtol=1e-12):
        raise ValueError("Covariances must be symmetric.")
    factors = np.linalg.cholesky(covariances)
    return means, covariances, factors


def score_mixture(means, covariances, observed):
    """Score a specified joint lognormal mixture, using its median for point errors.

    Finite mixtures have finite means, but the underlying unbounded lognormal
    noise-SD model need not. Quantile forecasts avoid relying on that moment.
    """
    means, covariances, factors = validate_mixture(means, covariances)
    y = np.asarray(observed, dtype=float)
    if y.shape != (means.shape[1],) or not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Need one finite positive response per target.")
    variance = np.diagonal(covariances, axis1=1, axis2=2)
    bounds = np.array(
        [
            [
                np.exp(gaussian_mixture_quantile(means[:, j], variance[:, j], p))
                for j in range(len(y))
            ]
            for p in [0.05, 0.5, 0.95]
        ]
    )
    lower, point, upper = bounds
    log_y = np.log(y)
    residual = log_y - means
    marginal = -0.5 * (np.log(2 * np.pi * variance) + residual**2 / variance) - log_y
    joint = []
    for r, factor in zip(residual, factors):
        white = solve_triangular(factor, r, lower=True, check_finite=False)
        joint.append(
            -0.5
            * (len(y) * np.log(2 * np.pi) + 2 * np.log(factor.diagonal()).sum() + white @ white)
            - log_y.sum()
        )
    pointwise = pd.DataFrame(
        {
            "observed": y,
            "point": point,
            "lower": lower,
            "upper": upper,
            "squared_error": (y - point) ** 2,
            "mean_bias": y - point,
            "coverage": ((y >= lower) & (y <= upper)).astype(float),
            "mean_width": upper - lower,
            "interval_score": upper
            - lower
            + 20 * ((lower - y) * (y < lower) + (y - upper) * (y > upper)),
            "marginal_predictive_nll": -(logsumexp(marginal, axis=0) - np.log(len(means))),
        }
    )
    joint_nll = -(logsumexp(joint) - np.log(len(means))) / len(y)
    return pointwise, float(joint_nll)


def aggregate_units(scores):
    """Equal-weight units, preserving pairing and horizons; no iid coverage CI."""
    keys = ["model", "unit_id", "horizon"]
    values = [
        "squared_error",
        "mean_bias",
        "coverage",
        "mean_width",
        "interval_score",
        "marginal_predictive_nll",
    ]
    if scores.empty or scores[keys + values].isna().any().any():
        raise ValueError("Need nonempty complete score rows.")
    if not np.isfinite(scores[values].to_numpy()).all():
        raise ValueError("Scores must be finite.")
    models = set(scores.model)
    identity = ["model", "unit_id", "horizon", "origin", "trial"]
    if scores[identity].isna().any().any() or scores.duplicated(identity).any():
        raise ValueError("Each model may score a target once within its horizon summary.")
    for _, group in scores.groupby("horizon"):
        if set(group.model) != models:
            raise ValueError("Every horizon must contain the same candidate models.")
        targets = [frozenset(zip(g.unit_id, g.origin, g.trial)) for _, g in group.groupby("model")]
        if len(set(targets)) != 1:
            raise ValueError("Candidate models must score identical targets.")
    # Unequal numbers of observations within units must not change their
    # weight in a comparison across independently collected units.
    per_unit = scores.groupby(keys, as_index=False)[values].mean()
    membership = per_unit.groupby(["model", "horizon"]).unit_id.agg(frozenset)
    for _, group in membership.groupby(level="horizon"):
        if len(set(group)) != 1:
            raise ValueError("Models must be compared on the same units at each horizon.")
    summary = per_unit.groupby(["model", "horizon"], as_index=False)[values].mean()
    counts = per_unit.groupby(["model", "horizon"]).size().to_numpy()
    summary["independent_units"] = counts
    summary["rmse"] = np.sqrt(summary.pop("squared_error"))
    return per_unit, summary


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_plan(plan_path, root=Path("."), inspect_prefixes=True):
    """Read declared inputs only, reporting unresolved experimental facts."""
    root = Path(root)
    plan = json.loads(Path(plan_path).read_text())
    metadata_path = root / plan["reference_metadata"]
    metadata = json.loads(metadata_path.read_text())
    missing = []
    input_missing = []
    for name in ["origins", "horizons"]:
        values = np.asarray(plan[name], dtype=float)
        if (
            values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError(f"{name} must be finite, nonempty, and strictly increasing.")
    if min(plan["origins"]) < 0 or min(plan["horizons"]) <= 0:
        raise ValueError("Need nonnegative origins and positive horizons.")
    if plan["time_scale"] != 12 or plan["interval_level"] != 0.9:
        raise ValueError("This protocol retains the fixed 12-unit time scale and 90% intervals.")
    if plan.get("point_forecast") != "posterior_predictive_median":
        raise ValueError("Point forecasts must use the predictive median under these SD priors.")
    if (
        set(plan["models"])
        != {"independent_ou", "shared_ou", "independent_matern32", "shared_matern32"}
        or len(plan["models"]) != 4
    ):
        raise ValueError("Protocol must specify the four fixed structural candidates.")
    for key in [
        "time_unit",
        "response_name",
        "response_unit",
        "experimental_unit_definition",
        "trial_to_experimental_unit",
        "shared_conditions",
        "confirmation_source",
    ]:
        if not metadata.get(key):
            missing.append("reference_metadata." + key)
    if not plan.get("independence_confirmed_by"):
        missing.append("independence_confirmed_by")
    if not plan.get("validation_units"):
        missing.append("validation_units")
    original = root / metadata["path"]
    if sha256(original) != metadata["sha256"]:
        raise ValueError("Development CSV changed since its metadata record.")
    units = []
    seen = set()
    for unit in plan.get("validation_units", []):
        name = unit["unit_id"]
        if name in seen:
            raise ValueError("Validation unit IDs must be unique.")
        seen.add(name)
        for field in ["provenance", "independence_basis", "outcomes_uninspected_declaration"]:
            if not unit.get(field):
                missing.append(f"validation_units.{name}.{field}")
        for field in ["time_unit", "response_unit"]:
            if not unit.get(field):
                missing.append(f"validation_units.{name}.{field}")
            elif metadata.get(field) and unit[field] != metadata[field]:
                raise ValueError("Convert new data to the confirmed reference units before use.")
        prefixes = unit.get("prefixes", {})
        if not inspect_prefixes:
            continue
        previous = None
        for origin in plan["origins"]:
            key = str(int(origin)) if float(origin).is_integer() else str(origin)
            if key not in prefixes:
                input_missing.append(f"validation_units.{name}.prefixes.{key}")
                continue
            path = root / prefixes[key]
            if not path.is_file():
                input_missing.append(str(path))
                continue
            frame = pd.read_csv(path)
            if "time" not in frame or len(frame) < 2:
                raise ValueError("Prefix must have time and at least two observations.")
            numeric = frame.to_numpy(dtype=float)
            if not np.isfinite(numeric).all() or np.any(frame.iloc[:, 1:].to_numpy() <= 0):
                raise ValueError("Prefix responses must be finite and positive.")
            if (
                frame.columns[0] != "time"
                or len(frame.columns) < 2
                or frame.columns.duplicated().any()
            ):
                raise ValueError("Prefix must start with time and contain distinct trial columns.")
            if (
                np.any(np.diff(frame.time) <= 0)
                or frame.time.iloc[0] != 0
                or frame.time.iloc[-1] != origin
            ):
                raise ValueError(
                    "Prefix must start at zero and end at its origin, with no future rows."
                )
            # Exact duplicate development measurements are not new validation
            # evidence, even when copied to a differently named file.
            from kefir_models.ode_fit import read_trial_csv, validate_frame

            development, columns = validate_frame(read_trial_csv(original))
            old = development.loc[development.time <= origin, ["time", *columns]].to_numpy()
            if old.shape == numeric.shape and np.array_equal(old, numeric):
                raise ValueError(
                    "A development-data prefix cannot be an independent validation unit."
                )
            if previous is not None:
                old_columns, old_numeric = previous
                if old_columns != list(frame.columns) or not np.array_equal(
                    old_numeric, numeric[: len(old_numeric)]
                ):
                    raise ValueError(
                        "Successive prefixes must preserve earlier observations and trial order."
                    )
            previous = (list(frame.columns), numeric)
            units.append(
                {"unit_id": name, "origin": origin, "prefix": prefixes[key], "sha256": sha256(path)}
            )
    return {
        "status": "awaiting_experimental_metadata"
        if missing
        else "protocol_ready_inputs_unchecked"
        if not inspect_prefixes
        else ("protocol_ready_inputs_pending" if input_missing else "ready_for_prefix_forecasts"),
        "protocol_ready_to_freeze": not missing,
        "missing": missing,
        "missing_prefix_inputs": input_missing,
        "prefix_inputs_inspected": inspect_prefixes,
        "development_sha256": sha256(original),
        "validated_prefixes": units,
        "plan_sha256": sha256(plan_path),
        "experimental_independence": (
            "requires documentary evidence; not established by file hashes"
        ),
    }


def freeze_protocol(plan_path, destination, root=Path(".")):
    """Snapshot a declared protocol before outcomes; future prefixes may be absent.

    This local record provides change detection, not external preregistration
    or proof of when an analyst first saw data. No target outcomes are read.
    """
    report = audit_plan(plan_path, root, inspect_prefixes=False)
    if not report["protocol_ready_to_freeze"]:
        raise ValueError("Experimental metadata and independent-unit declarations are incomplete.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(plan_path).read_text())
    shutil.copyfile(plan_path, destination / "plan.json")
    shutil.copyfile(
        Path(root) / plan["reference_metadata"], destination / "reference_metadata.json"
    )
    source_root = Path(__file__).resolve().parents[2]
    sources = [
        "src/kefir_models/" + name
        for name in [
            "bayesian_observation.py",
            "structural_discrepancy.py",
            "mcmc.py",
            "posterior_prediction.py",
            "prospective_validation.py",
            "ode_fit.py",
        ]
    ]
    sources += [
        "scripts/investigate_bayesian_observation.py",
        "scripts/investigate_structural_discrepancy.py",
    ]
    hashes = {name: sha256(source_root / name) for name in sources}
    for name in sources:
        target = destination / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: sha256(destination / name)
            for name in ["plan.json", "reference_metadata.json", *("source/" + n for n in sources)]
        },
        "model_source_sha256": hashes,
        "status": "protocol_frozen_validation_not_performed",
        "limitations": (
            "Local hash record, not external preregistration. "
            "New experimental independence is an attestation requiring evidence."
        ),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_protocol(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError("Frozen protocol changed; make a documented amendment.")
    for name, digest in manifest["model_source_sha256"].items():
        if sha256(Path(__file__).resolve().parents[2] / name) != digest:
            raise ValueError("Model source changed since the protocol was frozen.")
    return manifest
