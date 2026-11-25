# Report code review: issues for future iterations

Review date: 2026-11-25.
Update: 2026-11-25. REV-01 through REV-06 have been addressed and validated in
the first implementation iteration. The original review evidence below is
retained as history. REV-07 records the scientific calibration problem exposed
by the corrected rollout. See the implementation record at the end.
REV-08 implements explicit Bayesian observation models; REV-09 now records
the matched structural-discrepancy comparison. Forecast adequacy remains open.

## Scope and verified baseline

Reviewed the report rooted at `docs/00_main.tex` and its supporting modules:
`src/kefir_models/ode_fit.py`, `src/kefir_models/sde_compare.py`, and
`src/kefir_models/logistic_pinn_compare.py`, together with configurations,
plotting code, data, and tests.

Validation performed during the review:

- `python -m pytest -q`: 43 tests passed.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: 18 files already formatted.
- Refit the classical logistic model on CPU with 3000 epochs and learning
  rate 0.05. Reproduced RMSE 1.4291334725, carrying capacity 47.8602027893,
  and original-time growth rate 0.0458676730.
- Checked the logistic simulator against the exact deterministic solution
  with diffusion set to zero; see REV-03.

Full Neural ODE, Neural SDE, and PINN training was not rerun. Passing tests
therefore does not establish reproduction of all reported neural results.
No scientific source files were changed during the review. The working tree
already contained a modified `docs/00_main.pdf`; preserve that user change.

## REV-01: Stochastic interpretation differs between report and code

Priority: high. Status: resolved in the first implementation iteration (Itô retained).

Evidence:

- `docs/04_nsde.tex` describes a Stratonovich Neural SDE and uses circle
  stochastic-integral notation.
- `docs/06_sto_log_pinn.tex` also uses circle notation for logistic noise.
- `sde_compare.py::transition_negative_log_likelihood` and
  `simulate_sde_paths` implement an Itô Euler-Maruyama model.
- `logistic_pinn_compare.py::transition_negative_log_likelihood` and
  `simulate_logistic_sde` likewise use the stated drift without a
  Stratonovich-to-Itô correction.

Impact: the reported equations and implemented stochastic processes differ.
For scalar diffusion coefficient b(x), the corresponding Itô drift of a
Stratonovich equation is a(x) + 0.5 b(x) b'(x). For b(x) = sigma x, the
correction is +0.5 sigma^2 x. For the neural diffusion it depends on the
state derivative of the diffusion network.

Next iteration: establish the intended interpretation, then align equations,
transition objective, simulator, and descriptions. Do not silently choose an
interpretation by changing notation alone.

Acceptance: equations and implementation describe the same process, with a
meaningful analytic-case check of the selected interpretation.

## REV-02: Carrying-capacity constraints confound the model comparison

Priority: high. Status: resolved in the first implementation iteration.

Evidence:

- `sde_compare.py::train_classical_logistic_model` constrains capacity above
  the largest initial observation plus a small margin.
- `logistic_pinn_compare.py::build_logistic_data` instead sets
  `minimum_capacity = maximum_observed + 1e-3` using every observation.
- The supplied data give the PINN bound 52.1642004. The reproduced classical
  optimum is 47.8602028 and is therefore inadmissible for either PINN.
- Reported PINN capacities, 52.3350 and 52.2809, are close to that bound.

Impact: the classical and PINN methods solve differently constrained inverse
problems. The explanation in `docs/08_numerics.tex` attributes the PINN fit
gap to its composite objective without addressing this constraint difference.
A noisy observation need not be an upper bound on the latent capacity.

Next iteration: justify a common admissible parameter domain for the intended
comparison, or explicitly study and report the constraint difference. Check
sensitivity to the capacity lower bound before interpreting capacities near it.

Acceptance: compare fits under matched constraints, report any estimates near
their bounds, and revise the explanation of the RMSE gap using those results.

## REV-03: Simulation accuracy depends on the plotting grid

Priority: high. Status: resolved in the first implementation iteration.

Evidence:

- `logistic_pinn_compare.py::simulate_logistic_sde` takes exactly one
  Euler-Maruyama step between consecutive requested output times.
- `main` calls it separately on the 15 observation times and on a dense
  plotting grid, with different seeds.
- Metrics use the observation-grid simulation (12 original-time units per
  step). Dense plots default to 300 points (about 0.562 units per step).

Impact: table and plot use different discretizations as well as different
Monte Carlo samples. Changing `--plot-points` changes the simulated process
approximation rather than only its display resolution. The Neural SDE also
uses one step per observation interval and needs a discretization check.

Diagnostic performed with the supplied initial observations, r = 0.036806,
K = 52.2809, and sigma = 0:

- Compare the simulator with the exact logistic solution
  `K / (1 + (K / x0 - 1) * exp(-r * (t - t0)))`.
- At observation times, the 12-unit simulation has maximum absolute error
  0.9203894.
- A uniform grid of `299 * 14 + 1` points over [0, 168], sampled every 299
  steps, reduces that error to 0.0028441.
- Against the observations, the coarse and refined deterministic simulations
  have RMSE 3.3933507 and 2.9305715, respectively.

These are zero-noise diagnostics, not replacements for the reported
stochastic-model metrics and not evidence of full stochastic reproduction.

Next iteration: separate internal integration step size from requested output
times; obtain metric and plot summaries from a consistent simulation. Check
time-step convergence, Monte Carlo error, and the effect of positivity clipping.

Acceptance: deterministic-limit accuracy is verified; stochastic summaries
stabilize under refinement; changing display resolution does not materially
change predictions at shared times.

## REV-04: The documented pipeline does not regenerate the complete table

Priority: medium. Status: resolved in the first implementation iteration.

Evidence:

- `logistic_pinn_compare.py::build_metrics_frame` emits RMSE and R-squared,
  but does not calculate AIC/BIC or the corresponding residual variance.
- `docs/08_numerics.tex` contains literal PINN AIC/BIC values with no
  documented generation step in the reviewed pipeline.
- In `docs/09_reproducibility.tex`, the combined plotting command lacks a
  continuation backslash after the neural comparison CSV path. The isolated
  backslash on the following line does not continue the preceding command.

Impact: running the documented commands cannot regenerate every reported
table field, and copying the combined plotting command into a shell fails.

Next iteration: provide a documented table-generation step for every reported
metric, correct the command, and record the effective configuration and
environment needed to identify a report run. Keep training outputs and the
curated figures embedded by the report traceable to the same run.

Acceptance: a clean run produces all table fields and figures through
documented commands, with a comparison against the report's displayed values.

## REV-05: Residual AIC/BIC require narrower interpretation

Priority: medium. Status: resolved as an interpretation correction; no forecasting claim added.

Evidence:

- `sde_compare.py::gaussian_information_criteria` evaluates an independent,
  common-variance Gaussian residual score at the fitted mean predictions.
- Neural SDE training maximizes an approximate transition likelihood; PINN
  training optimizes weighted data, physics, and possibly transition terms.
  Those objectives do not maximize this residual likelihood over all weights.
- The score counts every trainable network parameter, with counts far above
  the 45 observed scalar responses.
- `docs/08_numerics.tex` already includes a useful caveat, but also describes
  the classical model as strongly preferred by these criteria.

Impact: these numbers are a particular parameter-penalized residual diagnostic;
they should not establish a definitive likelihood-based ranking of the
different stochastic and deterministic mechanisms.

Next iteration: retain explicit qualification wherever the ranking is stated,
identify the likelihood and parameter-count convention for each metric, and
separate descriptive fit from predictive validation. The report proposes
time-based holdouts, but the reviewed pipeline does not implement that evaluation.

Acceptance: conclusions distinguish in-sample fit, residual complexity scores,
transition likelihood, and any actually executed forecasting evaluation.

## REV-06: Tests do not validate the PINN scientific pipeline

Priority: medium. Status: resolved in the first implementation iteration.

Evidence: `tests/test_logistic_pinn_compare.py` contains only a CLI help-text
test. Existing passing tests do not cover the PINN physics residual,
parameter recovery, stochastic simulation accuracy, or report-table generation.

Next iteration: add focused scientific checks alongside the corresponding
fixes, especially the normalized-time chain rule, the deterministic logistic
limit, and consistent predictions at shared output times. Use small analytic
or synthetic cases; full training should not become a routine unit test.

Acceptance: tests can detect a wrong time-scale factor, an inaccurate
deterministic-limit simulator, and missing required generated metric columns.

## Suggested resumption order

1. Resolve REV-01 and the parameter-domain choice in REV-02.
2. Correct and validate simulation resolution in REV-03.
3. Complete report generation in REV-04 and qualify conclusions in REV-05.
4. Add the targeted checks in REV-06 with each relevant change.
5. Rerun affected experiments and regenerate the report after verification.

For each issue, record the decision, changed files, validation evidence, and
any regenerated results before changing its status to resolved. Recheck these
findings against current sources when resuming; file locations can move.


## First implementation iteration: decisions and validation

Date: 2026-11-25. The user authorized working through the review issues.
`AGENTS.md` now records numerical-analysis, stochastic-modeling, and Bayesian
inverse-problem standards, along with the requirement to consult this log.

- REV-01: retained the implemented Itô model as the working convention and
  explicitly corrected `docs/04_nsde.tex` and `docs/06_sto_log_pinn.tex`.
  A geometric-Brownian analytic-moment test distinguishes the implemented
  Itô drift from an uncorrected Stratonovich interpretation. This is a
  documented modeling decision, not a claim that the experiment selects Itô.
- REV-02: `build_logistic_data` now uses the same initial-observation-based
  capacity bound as the classical model. With fixed seed and epoch budgets,
  `results/validation/capacity_sensitivity.csv` compares the old and matched
  bounds. Deterministic PINN RMSE changes from 2.809979 to 1.603741; stochastic
  PINN drift RMSE changes from 2.838399 to 1.435671. These sensitivity runs
  share the revised initialization, so their old-bound values differ slightly
  from the historical report. The new fitted capacities are 49.099129 and
  48.238846, comfortably above the matched bound 14.870930. Fixed-epoch
  optimization is not certified to have reached a global optimum.
- REV-03: `src/kefir_models/numerics.py` supplies fixed-grid Euler-Maruyama
  simulation. Internal steps depend on endpoints and maximum step, not output
  resolution. Output values linearly interpolate that approximation.
  Observation and plotting runs use the same seed, so shared-time samples
  are unchanged when output points are added. The logistic projection count
  is recorded; it is zero in the reference and refinement runs. Tests check
  the exact deterministic logistic limit and output-grid invariance.
- REV-04: PINN metric CSVs now include all residual-score fields.
  `report_tables.py` generates LaTeX tables, numerical macros, and the combined
  metrics CSV. `scripts/reproduce_report.py` runs the fitting and plotting
  pipeline, generates the tables, and synchronizes the site's summary and
  figures. Effective arguments, versions, and data/source hashes are saved
  with each fit. The broken shell command was replaced with a complete
  reproduction procedure in `docs/09_reproducibility.tex`.
- REV-05: the report and model documentation now describe AIC/BIC as residual
  complexity scores. The variance is profiled conditional on fitted means;
  these means are not all fitted by maximizing that residual likelihood.
  Process-noise intervals are distinguished from Bayesian credible intervals,
  parameter uncertainty, and measurement noise. Proposed time-based validation
  is explicitly distinguished from executed experiments.
- REV-06: the unit suite now has 55 passing cases, including the normalized-time
  chain rule, capacity-domain admissibility, synthetic classical parameter
  recovery, deterministic-limit convergence, Itô geometric-Brownian moments,
  output-grid invariance, projection counting, invalid numerical settings,
  and complete PINN residual-score columns. Ruff lint and formatting pass.

All five model families were refitted using the reference CPU settings and
seed 7. The reference RMSEs are:

| Model | RMSE |
| --- | ---: |
| Classical logistic ODE | 1.429133 |
| Neural ODE | 1.028106 |
| Neural SDE | 2.476926 |
| Deterministic logistic PINN | 1.603741 |
| Stochastic logistic PINN drift | 1.435671 |
| Stochastic logistic SDE mean | 1.433181 |

The reference Neural SDE coverage is 22/45 (0.488889); logistic SDE coverage
is 41/45 (0.911111). These are conditional, in-sample diagnostics, including
fixed initial observations. The numerical tables are generated from the
saved outputs under `results/`, not copied from this log.

The report compiled to a separate 16-page review PDF at
`outputs/review_2026-11-25/report-review.pdf`. All pages were rendered for visual
inspection; the final LaTeX log has no warnings. The existing
`docs/00_main.pdf` was preserved. The documentation site also passed
`python -m mkdocs build --strict` with an isolated temporary site directory.

`results/validation/simulation_refinement.csv` records 3000 paths, seeds
8, 9, and 10, three maximum steps per model, and mean Monte Carlo standard
errors. Neural maximum normalized steps are 0.002, 0.001, and 0.0005;
logistic original-time steps are 0.1, 0.05, and 0.025. Across-seed average
coverage is 0.503704 at all three neural resolutions and approximately
0.896296--0.903704 for the logistic model. Neural average RMSE ranges from
2.486741 to 2.508892; logistic average RMSE ranges from 1.437921 to 1.440232.
This supports the observed calibration discrepancy across these finite
checks; it is not a proof of asymptotic convergence or forecast calibration.

To reproduce these diagnostics after fitting:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/check_simulation_refinement.py --capacity-sensitivity
```

## REV-07: Neural SDE continuous-time rollout is poorly calibrated

Priority: high. Status: investigated; transition mismatch diagnosed, forecast
calibration still open. See `docs/rev07-calibration.md` and `results/rev07/`.

The historical one-step-per-observation rollout gave 0.9333 coverage. The
refined reference rollout gives 0.488889, and three seeds across three fine
step sizes give about 0.50 coverage. Therefore the earlier high coverage
cannot be used as evidence of a calibrated continuous-time diffusion.

The model was fitted using a single-step Euler Gaussian approximation over
each observation interval. Refining a forward simulation does not correct
that likelihood approximation or establish the adequacy of the learned drift
and diffusion. The experiments identify a discrepancy; they do not uniquely
identify its cause.

Next scientific iteration:

1. Compare held-out transition and trajectory predictions using a split by
   time, fitting normalization only on training observations.
2. Assess sensitivity to the transition approximation, model size, regularization,
   and observation-error assumptions before selecting a more expensive
   likelihood or latent-process inference method.
3. Distinguish numerical and Monte Carlo error from statistical uncertainty.
   Any Bayesian extension must specify priors and the observation likelihood,
   and validate posterior computation and predictive calibration.

Acceptance: explain the transition-versus-rollout discrepancy with controlled
experiments, then establish calibrated predictive behavior under a stated
observation model and time-respecting validation. Do not inflate diffusion
merely to obtain nominal in-sample coverage.

### REV-07 investigation update, 2026-11-25

Completed a frozen-coefficient conditional simulation check, 40 neural fits
across five fixed candidate settings, four training cutoffs (including the
full-data fit), and two training seeds. Each neural fit has three independent
simulation seeds. Added an observation-noise logistic comparator and tests for
training-only scaling, moment propagation, and gradients; 62 tests pass.

The original checkpoint's conditional coverage drops from 0.9365 with one
Euler step to 0.5079 with 64 substeps. The refined transition experiment
substantially improves in-sample consistency. A compact refined candidate
achieves 0.9444 full-data rollout coverage, but only 0.4583 descriptive
held-out coverage across the overlapping time splits. All candidates
undercover on the tested forecasts; none is promoted to the reference pipeline.

The implementation and controlled numerical investigation are complete for
this iteration. The scientific acceptance criterion remains unmet. This
investigation motivated the observation/latent-process and parameter-uncertainty
extension recorded under REV-08 below.

## REV-08: Explicit observation model and parameter uncertainty

Priority: high. Status: implemented and computationally validated as an
exploratory model comparison. Predictive adequacy remains open under REV-07.

Implemented a logistic population median with uncertain initial value,
capacity, and rate; stationary OU deviations in log response; and independent
log-scale measurement error. A Bayesian observation-only comparator uses the
same priors on its common parameters. Exact Gaussian covariance likelihoods
marginalize the latent OU states; an independent Kalman calculation checks
them. Priors are proper, declared, fixed across cutoffs, and analyzed under
half/double noise-prior medians. Physical units remain unconfirmed, so priors
and rate conversions explicitly refer to original numerical units.

Four chains per case use adaptation during warmup only. All 16 combinations
of model, prior, and cutoff pass rank/folded split R-hat below 1.01 and bulk/
tail ESS at least 400. Retained lengths range from 12000 to 108000 per chain.
The largest R-hat is 1.0070, minimum bulk ESS 641, and minimum tail ESS 443.5.
These are computation checks, not proof of model adequacy. Sampler, exact
likelihood, conditional variance, and analytic prediction tests pass.

Reference OU held-out coverage is 29/36 = 0.8056 across overlapping forecasts,
versus 26/36 = 0.7222 for the Bayesian observation-only comparator. Fixing the
OU coefficients at posterior marginal medians gives 25/36 = 0.6944. OU
noise-prior sensitivity ranges from 28/36 to 30/36 coverage. The 36 entries
reuse only 15 distinct responses and do not establish independent-experiment
calibration. Noise decomposition and correlation time remain weakly identified.

Rao-Blackwellized mixture CDFs replace additional simulations of observation
error when computing predictive quantiles. At cutoff 132, doubling the
posterior-integration subset from 12000 to 24000 leaves coverage unchanged;
the full 6000/12000/24000 check has endpoint shifts at most about 0.012 units.
All seven uncovered reference OU forecast entries are above the upper bounds,
and mean underprediction increases with later cutoffs. This motivates the next
scientific point: structural discrepancy and additional-trajectory validation,
without claiming that more noise or a converged sampler resolves the bias.

The LaTeX report now integrates REV-07 and REV-08 in dedicated sections,
including equations, generated tables, predictive plots, priors, diagnostics,
and limitations. See `docs/bayesian-observation.md`, `results/bayesian/`,
`src/kefir_models/bayesian_observation.py`, and
`python scripts/run_bayesian_iteration.py` for continuation and reproduction.

Final verification for this iteration: 77 tests pass; Ruff lint/format and the
strict documentation build pass. The updated report compiles without LaTeX
warnings to a separate 25-page PDF, and all pages were rendered and visually
checked. The original `docs/00_main.pdf` remains unchanged. The deliverable is
`outputs/review_2026-11-25/report-updated-bayesian.pdf`; source/artifact hashes
and validation counts are recorded in `results/bayesian/report_manifest.json`.

## REV-09: Structural discrepancy and late underprediction

Priority: high. Status: controlled comparison implemented and computationally
validated; independent-experiment validation and predictive adequacy remain open.

Crossed independent versus shared log-response discrepancy with OU versus
Matérn-3/2 kernels. Matched stationary variance and e-fold correlation time
make the same six-coordinate priors comparable across kernels. Shared
discrepancy replaces the independent process and preserves cross-trial
covariance in likelihoods and joint forecasts. It does not add trial random
effects or establish experimental independence. Both kernels remain stationary
with zero mean, so neither permanently changes the logistic asymptote.

Completed 56 posterior cases: 44 new fits and 12 verified frozen independent
OU cases. All pass the declared R-hat/ESS gates. New fits retain 36000 or
108000 draws per chain after 6000 warmup iterations. Across all cases the
largest R-hat is 1.0070, minimum bulk ESS 641, and minimum tail ESS 443.5.
Dense multivariate-normal likelihood/conditioning oracles verify the shared
reduction and joint predictive densities. Stationary Lyapunov and matrix-
exponential checks independently validate the OU/Matérn SDE kernels. Frozen
OU forecasts and scores are reproduced before comparison.

Added distinct one-step targets at times 120, 132, 144, 156, and 168. Their
15 scalar responses are each scored once; `one_step_target_audit.csv` checks
that construction. They still belong to the same three trial trajectories.
The original multi-time summary retains 36 overlapping forecast entries.
Interval scores penalize unnecessary width; marginal and joint predictive
densities retain posterior parameter uncertainty and their appropriate
dependence assumptions.

Reference shared OU multi-time coverage is 35/36 (0.9722), versus independent
OU 29/36 (0.8056). Width increases from 5.857 to 9.138; interval score improves
from 10.164 to 9.371. Mean underprediction persists at 1.373 versus 1.648.
For distinct one-step targets, coverage is 14/15 versus 13/15, and interval
scores are 7.489 versus 8.224. Marginal NLL improves under sharing; joint
per-entry NLL instead favors independent OU under both summaries. Hence
these improvements do not validate the stronger cross-trial dependence.
Independent Matérn results are close to independent OU; smoother dynamics
alone do not resolve the observed departure. No candidate is promoted.

The full-data shared OU capacity credible interval is approximately
[45.555, 52.107], compared with [47.177, 49.050] for independent OU. Greater
capacity uncertainty is consistent with confounding between a common model
departure and the population growth law. All three noise-prior settings are
reported, without selecting one by coverage. At cutoff 132 the three new
alternatives retain unchanged coverage under 6000/12000/24000 posterior
integration draws, with maximum endpoint shift about 0.0442 response units
from the 6000-draw calculation.

Changed implementation: `src/kefir_models/structural_discrepancy.py`, the
predictive joint-density dispatch in `scripts/investigate_bayesian_observation.py`,
and `scripts/{investigate_structural_discrepancy,run_structural_iteration,
summarize_structural_discrepancy,build_structural_tables}.py`. Experiment
artifacts and provenance are in `results/structural/`. Continuation guidance
is in `docs/structural-discrepancy.md`; the LaTeX report includes
`docs/12_structural_discrepancy.tex` and generated tables/interpretation.

Validation: 90 tests pass; Ruff lint/format and strict MkDocs build pass.
The separate updated PDF is `outputs/review_2026-11-25/report-structural-discrepancy.pdf`.
The original user PDF is preserved. Report hashes and final visual-validation
evidence are recorded in `results/structural/report_manifest.json`.

Next acceptance criteria: confirm measurement units and shared experimental
conditions; freeze priors, candidate models, forecast horizons, and scoring
rules before inspecting additional independently collected trajectories.
Report validation by trajectory/horizon and estimate calibration uncertainty
across independent experimental units when sufficient replication exists.
Neither successful computation nor retrospective 90% coverage closes REV-07.

## REV-10: Independent-unit validation and experimental metadata

Priority: high. Status: protocol and computational preparation implemented;
experimental metadata and new-unit validation remain pending.

The only available experimental CSV is the already inspected development
data: 15 time points, three trial series, 45 scalar responses. The CSV does
not confirm physical units, the experimental unit, or common batch/instrument
effects. Created `data/metadata/reference.json` with the verified data hash
and explicit null entries for unconfirmed facts. No new independent units
are declared and no experimental facts were inferred from figure labels.

Prepared `configs/prospective_validation.json`: four fixed candidates,
existing priors and noise sensitivities, declared MCMC gates, draft origins
108/120/132/144/156 and horizon 12, central 90% intervals, primary interval
score, and equal weights across independent experimental units. The point
forecast is the predictive median following REV-11. These numerical choices
remain a draft until units and the experimental schedule are confirmed.

Implemented a readiness audit, local non-overwriting protocol snapshot and
verification, outcome-free joint lognormal-mixture forecasts, separate
response scoring, and paired target/experimental-unit aggregation. The
protocol can be frozen before future prefix files exist; its snapshot
operation does not read the new observations. Snapshots retain source copies
and hashes but do not establish external preregistration or outcome-access
timing. Prefix audits reject future rows, altered earlier values/trial order,
and exact development prefixes copied under another filename. Hashes cannot
prove experimental independence.

The current readiness report is `awaiting_experimental_metadata` with nine
unresolved declarations and zero declared new units. A complete automated
new-data fitting and forecast-archive runner is not implemented. The APIs
accept validated prefix posterior fits once those inputs exist. No new-data
performance or confidence interval is reported. Report weights average within
each independent unit and then across units, retaining the forecast horizon;
they do not treat longitudinal response rows as independent experiments.

Implementation and continuation: `src/kefir_models/prospective_validation.py`,
`scripts/check_prospective_validation.py`, `scripts/build_prospective_tables.py`,
`docs/prospective-validation.md`, and `docs/13_prospective_validation.tex`.
Readiness and generated CSVs are under `results/prospective/`.
Synthetic/Gaussian-oracle tests validate joint density normalization, median
point forecasts, forecast-time restrictions, protocol change detection,
development-data reuse checks, and unequal unit-size weighting. They are
computational tests, not independent kefir validation.

Acceptance still requires experimental confirmation of units/grouping,
additional independently collected trajectories, a frozen analysis before
target outcome inspection, and a fully documented new-data analysis.

## REV-11: Response-scale predictive mean does not exist

Priority: high. Status: mathematical diagnosis complete; new prospective
point forecasts use predictive medians. Finite-mean prior redesign is open
if response-scale posterior expectations are required.

With `z_b = log b` normally distributed a priori, fixing other parameters in
a compact interior set gives `log L(z_b) = -N*z_b + O(1)` as `z_b` grows,
where N is the training scalar-response count. A new observation has
`log E[Y_* | z_b, theta, D] = exp(2*z_b)/2 + O(1)`. The log first-moment
integrand is therefore
`exp(2*z_b)/2 - (z_b-mu_b)^2/(2*d_b^2) - N*z_b + O(1)` and tends to
positive infinity. Uniform bounds on a positive-volume compact set of other
parameters establish divergence of the joint-posterior predictive mean.
The argument applies to both sharing assumptions and both temporal kernels;
changing the noise-prior median or adding finitely many observations cannot
remove the divergence.

Earlier finite posterior-subset mean curves and their RMSE/bias scores are
finite-mixture numerical diagnostics. R-hat/ESS and stable finite averages
cannot establish the existence of the full predictive expectation. The
earlier priors and saved fits are preserved; their interpretation is now
qualified in the report and Bayesian/structural documentation. Predictive
quantiles and densities remain well-defined distributional summaries.

The new prospective scorer uses the posterior predictive median for point
errors, including descriptive median-forecast RMSE. It does not claim finite
posterior expected quadratic risk. It retains full joint predictive covariance
and integrates over parameters in its CDF/density calculations. A physical
noise bound has not been invented to repair a prior in unconfirmed units.

`scripts/check_predictive_moments.py` writes a log-scale tail check for all
four candidates to `results/prospective/predictive_moment_tail.csv`. It verifies
the polynomial likelihood decay numerically without overflowing a response
mean. This finite-grid check supports the analytic proof; it is not itself
proof of divergence. Synthetic tests independently check the likelihood
asymptotic.

Current iteration verification: 102 tests pass; Ruff lint/format and strict
MkDocs checks pass. The updated LaTeX report includes the readiness protocol,
unit-level aggregation, and predictive-moment derivation. The separate PDF
is `outputs/review_2026-11-25/report-prospective-validation.pdf`; the original
user PDF and earlier review PDFs are preserved. Final artifact hashes and
visual checks are recorded in `results/prospective/report_manifest.json`.

## REV-12: A separate noise-prior family with finite response moments

Priority: high. Status: implemented, mathematically justified, and refitted.
This resolves the moment concern for the new family; historical priors and
their infinite response means are retained as previously qualified.
Forecast adequacy and independent experimental validation remain open.

For each log SD, the new normalized prior is
`Normal(z; mu_eta, 1) * exp(-eta*exp(2*z)) / Z_eta`. Both latent amplitude
and measurement SD receive the penalty. It leaves positive SD support
unbounded. One-dimensional quadrature and a root solve calibrate the Gaussian
kernel location so that each SD median is 0.03. Kernel locations are not
means of the tempered log-SD prior. The 95th SD percentile changes from
0.15541 historically to 0.14834 for eta=2 and 0.13590 for eta=8; matching
medians does not preserve the whole prior shape.

The global posterior bound uses the measurement likelihood conditional on
latent states, bounded by `C(D)*b^(-N)`, before integrating the stationary
Gaussian discrepancy. With logistic median at most K, the q-th predictive
moment is bounded by a finite constant times
`E_G[K^q*b^(-N)*exp((q*q/2-eta)*(a*a+b*b))]`, where G is the product of
Gaussian kernel priors. This is finite for `eta >= q*q/2`, including the
boundary. Capacity positive moments and lognormal negative SD moments are
finite. The same bound establishes a proper posterior. Conversely the
large-b tail diverges for `q*q/2 > eta`. Thus eta=2 gives finite first and
second response moments; eta=8 gives finite fourth moments. Neither gives
all positive moments. The proof applies globally to both sharing assumptions
and both kernels without a bound on conditional interpolation weights.

Completed 40 refits: all four structural models at six cutoffs for eta=2
and four cutoffs for eta=8. Earlier posterior samples were not reused under
a different prior. All cases pass rank/folded split R-hat below 1.01 and
bulk/tail ESS at least 400, plus raw ESS at least 400 for monitored
conditional-mean integrands. Largest R-hat is 1.0059, minimum bulk ESS 928.8,
minimum tail ESS 423.4, and minimum raw conditional-mean ESS 948.2. The
largest estimated MCSE for those observables is 0.0717 response units.
Functional diagnostics use every eighth retained draw at first/last target
times. Retained lengths are 36000 per chain in 38 cases and 108000 in two,
following 6000 warmup iterations. These checks do not prove a Markov-chain
CLT or bound unobserved Monte Carlo tails.

Coverage is unchanged from the earlier reference comparison: multi-time
independent models 29/36, shared OU 35/36, shared Matérn 34/36; distinct
one-step targets have 13/15 coverage for independent models and 14/15 for
shared models. The stronger prior leaves the multi-time coverage unchanged.
Main-prior independent/shared OU interval scores are 10.092 and 9.449;
mean-forecast biases remain 1.642 and 1.370. Late underprediction persists.
Joint density scores still favor independent OU while marginal scores favor
shared OU. Small score changes are not a basis for fine model ranking or
for promoting a candidate. The repeated holdouts are not new experiments.

At cutoff 132, independent/shared OU integration checks with 6000, 12000,
and 24000 components give maximum mean shift 0.0136 and endpoint shift 0.0150
response units from the 6000-component calculation, with unchanged coverage.
Analytic prior normalization, median calibration, exponential-square moment
identities, exact likelihood preservation, and transformed-target density
tests provide independent implementation checks.

Implementation: `src/kefir_models/tempered_noise.py`,
`scripts/{investigate_tempered_noise,run_tempered_iteration,
summarize_tempered_noise,build_tempered_tables}.py`, and
`tests/test_tempered_noise.py`. Results, diagnostics, prior checks, parameter
summaries, and source/input hashes are under `results/tempered/`.
Continuation notes are in `docs/tempered-noise.md`; the report includes
`docs/14_tempered_noise.tex` and generated CSV-based tables and interpretation.

Validation: 118 tests pass; Ruff lint/format and strict MkDocs checks pass.
The updated PDF is saved separately as
`outputs/review_2026-11-25/report-tempered-noise.pdf`; original and earlier
review PDFs are preserved. Final artifact hashes and visual checks are in
`results/tempered/report_manifest.json`.

Remaining acceptance work: confirm experimental metadata, repeat noise-median
sensitivity for this family if it is adopted, explicitly amend the prospective
protocol before changing its priors, and validate on independently collected
trajectories. The current prospective draft still uses its earlier priors
and predictive-median point forecast. REV-07 and REV-10 remain open.
