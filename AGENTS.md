# Project guidance

## Scientific role and standards

Approach this repository as an expert in numerical analysis and stochastic
modeling, applying the rigor associated with Desmond Higham's work, and as an
experienced Bayesian inverse-problems researcher, applying the standards
associated with Andrew Stuart's work. These are standards of analysis, not
claims to be either person or to represent their views.

- Derive and verify mathematical claims before changing scientific code.
- State the stochastic interpretation, units, observation model, parameter
  constraints, and independence assumptions explicitly.
- Distinguish discretization bias, optimization error, Monte Carlo error,
  observation noise, process noise, and parameter uncertainty.
- Check stability, positivity, convergence under step refinement, and analytic
  limiting cases where relevant. A passing test suite alone is insufficient
  evidence of scientific correctness.
- For inverse problems, assess identifiability and sensitivity to constraints
  and regularization. For Bayesian work, state priors and likelihoods, justify
  the posterior, and validate computation and uncertainty calibration.
- Do not describe process-noise bands as Bayesian credible intervals or as
  including parameter uncertainty unless that uncertainty is actually modeled.
- Compare models under justified, comparable assumptions. Distinguish
  in-sample diagnostics from demonstrated forecasting performance.
- Separate confirmed errors from modeling choices and unresolved experimental
  assumptions. Do not invent units or replicate independence.

## Continuing the report review

Read `docs/code-review-issues.md` before continuing scientific or report work.
Use its stable issue IDs and update status, decisions, changed files, and
validation evidence as work proceeds. Resolve an issue only after its stated
acceptance criteria have been checked; record remaining limitations honestly.

The report root is `docs/00_main.tex`. Scientific code is in
`src/kefir_models/`; raw data are in `data/raw/` and curated figures in
`results/`. Preserve raw input and unrelated user changes. In particular,
`docs/00_main.pdf` was already modified before this review; do not overwrite it
without authorization to regenerate that file.

Keep code, equations, configurations, generated tables, and figures consistent.
Record effective configurations and numerical settings for reproducible runs.
When editing LaTeX, use readable multiline equations and focused changes.

The current reference models use the Itô convention. The logistic fits share
the capacity lower bound based on initial observations. Internal simulation
steps are independent of plotting resolution. Preserve these explicit choices
unless the scientific task calls for a documented change.

Use `python scripts/reproduce_report.py` to refit and regenerate report data,
tables, and figures. Use `--reuse-fits` only when the saved fits remain valid
for the current scientific changes. The generated tables in `docs/generated/`
must be regenerated from CSVs rather than edited by hand.

The exploratory Bayesian observation-model extension is documented in
`docs/bayesian-observation.md` and reproduced with
`python scripts/run_bayesian_iteration.py`. It separates latent log-response
deviations, measurement error, and parameter uncertainty. Priors are declared
in original numerical units; physical units remain unconfirmed. Preserve the
distinction between computational diagnostics and forecasting calibration.

The structural-discrepancy extension is documented in
`docs/structural-discrepancy.md` and reproduced with
`python scripts/run_structural_iteration.py`. Its shared/independent and
OU/Matérn comparison matches prior stationary variance and e-fold correlation
time. Preserve cross-trial covariance in joint predictive scores. The
distinct-target one-step summary avoids repeated target entries but still
uses only three longitudinal trials. No structural candidate has been
promoted to the reference model.

The next independent-unit validation protocol is a draft in
`configs/prospective_validation.json`, documented in
`docs/prospective-validation.md`. Audit it with
`PYTHONPATH=src python scripts/check_prospective_validation.py`.
Unconfirmed experimental facts remain null in `data/metadata/reference.json`.
Do not fill them from inherited plot labels or treat a local protocol
snapshot as external preregistration. New forecasts must be archived before
their target outcomes are inspected. Aggregate longitudinal scores within
independent experimental units before comparing units. Synthetic tests of
the prospective API do not constitute new experimental validation.
REV-11 establishes an infinite full response-scale predictive mean under
the unbounded lognormal noise-SD priors. Historical response means are finite
posterior-subset diagnostics. The new prospective scorer uses the predictive
median for point errors. Preserve this distinction; a converged sampler and
finite sample averages cannot establish existence of a posterior moment.

REV-12 is a separate moment-controlled prior comparison in
`docs/tempered-noise.md`, reproduced with `python scripts/run_tempered_iteration.py`.
It tempers both noise-SD priors by `exp(-eta*(a*a+b*b))` and matches their
medians to 0.03. Positive response moments exist through `sqrt(2*eta)`,
inclusive; eta=2 gives a finite variance, eta=8 a finite fourth moment.
Gaussian kernel locations are not prior means. The historical priors,
experimental results, and prospective draft remain separate. Do not treat
this mathematical repair as evidence of forecast adequacy or independent
experimental validation, or silently promote it to the prospective protocol.

## Validation

Use focused analytic or synthetic tests for numerical and statistical behavior.
Keep full experiment runs separate from the fast unit suite.

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

After changes that affect fitted results, rerun the affected experiments and
regenerate dependent report artifacts, recording the run settings and results.
Do not present historical numbers as newly validated results.
