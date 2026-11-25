# Independent-trajectory validation: preparation and outstanding inputs

REV-10 prepares the next experiment after the [structural comparison](structural-discrepancy.md).
No new independent-trajectory validation has been performed. The checkout
contains only the original, already inspected CSV. Its physical units,
experimental unit definition, and shared conditions are undocumented.
The numerical scores from REV-07–09 remain exploratory results.

An additional moment audit (REV-11) establishes that the unbounded lognormal
noise-SD priors give an infinite full response-scale predictive mean. Earlier
finite posterior-subset mean curves are retained as finite-mixture numerical
forecasts; their finite values do not estimate a finite full-posterior mean.
The new point forecast is the predictive median. Intervals and density
scores retain their predictive-distribution interpretation.

## Data contract

`data/metadata/reference.json` records the original CSV hash and leaves
unconfirmed experimental facts as `null`. Confirm the measured quantity,
time and response units, experimental unit definition, mapping of trial
columns to those units, common batches/instruments/disturbances, and the
source of that information. Inherited plot labels are not evidence.

The draft `configs/prospective_validation.json` retains four models,
the existing six-coordinate priors, 12-unit time scale, reference noise
prior, fixed half/double sensitivities, four-chain computation gates,
and 12000 posterior integration components. These are proposed prospective
choices after the exploratory work, not a protocol already registered.
Confirm units before interpreting or freezing these numerical priors.

Each new `unit_id` must identify an independently collected experimental
unit supported by the experimental record. If flasks share a relevant
batch-level disturbance, their common batch must be treated as the unit
for cross-unit uncertainty assessment. File names and distinct columns do
not establish independence. No required number of independent units is
invented here; study size and desired precision still need experimental
planning. A single new unit supplies a descriptive replication, not a
population calibration guarantee.

Add a unit entry to `validation_units` only when its identity and provenance
are known. For example, the following describes the schema, not actual data:

```json
{
  "unit_id": "actual_independent_unit_id",
  "provenance": "experimental record and collection date",
  "independence_basis": "documented relationship to development and other units",
  "outcomes_uninspected_declaration": "who confirms the timing of outcome access",
  "time_unit": "confirmed reference time unit",
  "response_unit": "confirmed reference response unit",
  "prefixes": {}
}
```

Declare `independence_confirmed_by` with the responsible experimental source.
Future prefix paths may be absent when the protocol is frozen. A prefix is
a wide CSV with `time` first and one positive-response column per trial;
its first time is zero, times increase, and its last row is the forecast
origin. It must contain no later observations. Add paths as observations
become available, keyed by origin, e.g. `"108": "data/new/unit_prefix_108.csv"`.
Changes to the input-path registry should be logged separately from changes
to model, scoring, or experimental decisions. Retain the frozen plan; audit
a working copy when registering later prefixes. Prefix audits check that
later prefixes preserve earlier values and trial order. Exact copies of
development-data prefixes are rejected even under a new filename. Passing
these checks cannot prove experimental independence.

## Freeze the comparison before examining target outcomes

Run from the repository root:

```sh
PYTHONPATH=src python scripts/check_prospective_validation.py
```

The current report, `results/prospective/readiness.json`, says
`awaiting_experimental_metadata`. It lists the missing fields and declares
that the protocol is not ready to freeze. This is an incomplete-data state,
not a successful scientific validation.

Once the required experimental declarations are complete, create a new
snapshot before looking at the target outcomes:

```sh
PYTHONPATH=src python scripts/check_prospective_validation.py \
    --freeze results/prospective/frozen_protocol_v1
PYTHONPATH=src python scripts/check_prospective_validation.py \
    --verify results/prospective/frozen_protocol_v1
```

The freeze operation itself reads the protocol and experimental metadata,
and hashes the already inspected development CSV; it does not inspect new
prefix observations. It copies the plan, metadata, and relevant model and
inference sources, records their hashes and a local UTC timestamp, and
refuses to overwrite an existing snapshot. Verification checks the saved
copies and the current model sources. A local hash record provides change
detection, not external preregistration or proof of first outcome access.
Do not describe it as a prospective registration unless that separate
procedure actually took place.

## Forecast and reveal separately

The proposed primary task predicts one step (12 original time units) after
origins 108, 120, 132, 144, and 156. These origins and horizon are a draft
choice to confirm against the new experimental schedule before freezing.
New data must be converted to the confirmed reference units; changing the
time or response units without transforming the priors changes the inverse
problem. Amend the protocol explicitly for a different design.

At each origin, fit each of the four candidates using only that unit's
available prefix, with the frozen priors and sampler rules. This evaluates
within-trajectory forecasts on new experimental units; it does not transfer
a fixed fitted capacity to all future experiments or introduce hierarchical
pooling. Require the declared MCMC checks before scoring, and report failures
rather than silently omitting difficult units. Do not tune noise priors,
smoothness, or interval levels to the target outcomes.

The new `kefir_models.prospective_validation` API separates the two operations:

```python
# problem is a StructuralProblem containing the available training prefix.
# chains come from its validated four-chain posterior fit.
log_means, log_covariances = gaussian_forecast(
    problem, chains, future_times_in_problem_scale, count=12000
)
# Archive these moments, target identities, source/input hashes, diagnostics,
# units, and creation time before revealing the corresponding responses.

# Only after the frozen forecast is archived:
pointwise_scores, joint_nll = score_mixture(
    log_means, log_covariances, observed_responses
)
```

Mixture entries use time-major, then trial ordering. Every component retains
the full joint covariance across target times and trials. The archive can be
saved with `numpy.savez_compressed`; do not replace it after outcomes are
revealed. `gaussian_forecast` rejects target times at or before the last
training time. It has no target-response argument. `score_mixture` computes
central 90% observation intervals, a predictive median for point errors,
interval scores, response-scale marginal
density scores, and the joint mixture density including the log-transform
Jacobian. A joint mixture density integrates common parameter uncertainty;
it is not a product of separately integrated marginal densities.

These functions and protocol checks are preparation utilities. They do not
yet provide a complete automated acquisition, new-data posterior-fitting,
and forecast-archive runner. The existing experiment scripts remain tied to
the development study. End-to-end new-data validation must be run and its
provenance checked once the experimental inputs are supplied.

## Score independent experimental units

The primary score is the central 90% interval score; coverage, width, bias,
RMSE, and marginal/joint NLL are prespecified diagnostics. Retain joint
scores because the earlier shared models improved marginal scores while
giving worse joint density scores. The use of interval and density scores
follows [Gneiting and Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

For each model and horizon, average pointwise scores within each independent
unit first, then average the resulting unit scores with equal weights.
`aggregate_units` implements this rule. Its input combines pointwise scores
with `model`, `unit_id`, `horizon`, `origin`, and `trial`. It checks that models
have identical target identities and rejects duplicate target rows.
RMSE is the square root of the mean of unit-level mean squared errors,
not an average of the unit RMSEs. Joint NLL should be retained per forecast
block and averaged within and across units under the same hierarchy; the
pointwise aggregation function does not infer joint densities from marginals.

Report results by unit and horizon before any aggregate. Compare models with
paired unit-level score differences. If uncertainty intervals across units
are later computed, resample independent units and retain all their times,
trials, and model pairings together; do not bootstrap individual response
rows as though they were independent experiments. The current code does not
produce calibration confidence intervals or make a significance claim.

The numerical tests use synthetic fixtures and Gaussian density oracles;
they validate computation, not calibration on new kefir experiments.
REV-07 and REV-10 remain scientifically open until the missing experimental
facts and additional trajectories are supplied and analyzed.

## Predictive-moment audit

For the measurement SD, write `z = log(b)`. Its prior is normal. With N
training responses, holding other parameters in a compact interior set gives
`log L(z) = -N*z + O(1)` as z grows, while the log conditional response mean
is `exp(2*z)/2 + O(1)`. Thus the first-moment integrand has log asymptotic
`exp(2*z)/2 - (z-mu)^2/(2*sd^2) - N*z + O(1)`, which tends to infinity.
The bound is uniform on a positive-volume compact set of the other
parameters, establishing divergence also for the joint posterior.

This mathematical problem persists for arbitrarily many finite observations
and half/double noise-prior medians. It is not a failure of MCMC convergence.
Predictive quantiles remain finite and define the new point summaries. RMSE
of those median forecasts is a descriptive observed error, not a claim of
finite posterior expected squared loss. A prior redesign with justified
tail control is required if finite response means are scientifically needed;
an arbitrary noise bound is not introduced here.

```sh
PYTHONPATH=src python scripts/check_predictive_moments.py
```

The log-scale likelihood check writes
`results/prospective/predictive_moment_tail.csv`. The finite grid verifies
the likelihood asymptotic; the divergence conclusion comes from the
analytic tail argument in the report, not numerical extrapolation.
