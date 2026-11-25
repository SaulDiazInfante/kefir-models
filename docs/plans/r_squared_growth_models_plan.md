# Plan: $R^2$ for the deterministic and stochastic growth models

Status: plan, 25 November 2026. No code has been written yet.

## Goal

Compute the coefficient of determination $R^2$ for six fitted models on the
water kefir data (`data/raw/waterKefirTrialsReferece.csv`):

| Family | Deterministic (ODE) | Stochastic (SDE, multiplicative noise) |
|---|---|---|
| Logistic | $\dot y = r\,y(1-y)$ | $dY_t = r\,Y_t(1-Y_t)\,dt + \sigma Y_t\,dW_t$ |
| Gompertz | $\dot y = -r\,y\log y$ | $dY_t = -r\,Y_t\log Y_t\,dt + \sigma Y_t\,dW_t$ |
| von Bertalanffy | $\dot y = r(1-y)$ | $dY_t = r(1-Y_t)\,dt + \sigma(1-Y_t)\,dW_t$ |

Here $y = x/K$ with a given, fixed $K > 0$ (the manuscript currently uses
$K = \max_{i,j} x_i^{(j)} = 52.1632$), and results are reported on the
original scale $x = K y$. The data contain three independent flask runs
$j = 1, 2, 3$, observed at $t_i = 12\,i$ h, $i = 0, \dots, 14$
($\Delta = 12$ h), so $n = 45$ observations.

The definition must agree with the existing comparison criteria
(`docs/07_comparision_criteria.tex`, `compute_r_squared` in
`src/kefir_models/sde_compare.py`):

$$
R^2 = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}, \qquad
\mathrm{SSE} = \sum_{(i,j)\in\Omega} \bigl(x_i^{(j)} - \widehat x_i^{(j)}\bigr)^2, \qquad
\mathrm{SST} = \sum_{(i,j)\in\Omega} \bigl(x_i^{(j)} - \bar x\bigr)^2 .
$$

The only model-specific ingredient is the fitted value $\widehat x_i^{(j)}$.

## 1. What counts as the fitted value

An SDE does not produce a single curve, so "the $R^2$ of an SDE" needs a
definition. The plan computes two complementary versions for every model.

### 1a. Trajectory $R^2$ (main comparison)

Fitted value: the mean trajectory started at each run's first observation,

$$
\widehat x_i^{(j)} = K\,\mathbb E\bigl[Y_{t_i} \mid Y_0 = y_0^{(j)}\bigr]
\quad\text{(SDE)}, \qquad
\widehat x_i^{(j)} = K\,y\bigl(t_i;\,y_0^{(j)}\bigr)
\quad\text{(ODE)} .
$$

Exclude $i = 0$ from $\Omega$ (the fit reproduces it by construction), so
$|\Omega| = 42$. This matches how the manuscript figure compares fitted means
with the data.

### 1b. One-step-ahead $R^2$ (secondary, fairer to the SDEs)

Fitted value: the conditional mean one sampling interval ahead,

$$
\widehat x_i^{(j)} = K\,\mathbb E\bigl[Y_{t_i} \mid Y_{t_{i-1}} = y_{i-1}^{(j)}\bigr],
\qquad i = 1, \dots, 14 .
$$

For the ODEs, use the deterministic flow over one interval $\Delta$ from
$y_{i-1}^{(j)}$. This is the predictive quantity the SDE likelihoods are built
on, and it does not penalize a stochastic model for drifting away from one
observed path.

### Optional: median fitted value

For skewed transitions (Gompertz is lognormal), also report $R^2$ with the
conditional median as fitted value, as a robustness check.

## 2. Mean trajectories

Write $\phi_t = e^{-rt}$.

**Deterministic solutions.**

$$
y_{\mathrm L}(t) = \frac{y_0 e^{rt}}{1 - y_0 + y_0 e^{rt}}, \qquad
y_{\mathrm G}(t) = \exp\bigl(\phi_t \log y_0\bigr), \qquad
y_{\mathrm V}(t) = 1 - (1 - y_0)\,\phi_t .
$$

**Stochastic Gompertz (closed form).** $Z_t = \log Y_t$ is an
Ornstein–Uhlenbeck process, so $Y_t$ is lognormal with

$$
m(t) = \phi_t \log y_0 - \frac{\sigma^2}{2r}(1 - \phi_t), \qquad
v(t) = \frac{\sigma^2}{2r}\bigl(1 - \phi_t^2\bigr), \qquad
\mathbb E[Y_t] = \exp\bigl(m(t) + v(t)/2\bigr) .
$$

The median is $\exp(m(t))$. Unlike the ODE, the SDE mean depends on
$\sigma$.

**Stochastic von Bertalanffy (closed form).** $1 - Y_t$ is a geometric
Brownian motion with zero-mean noise, so

$$
\mathbb E[Y_t] = 1 - (1 - y_0)\,\phi_t ,
$$

which equals the ODE solution. With different estimates of $r$, the two $R^2$
values still differ.

**Stochastic logistic (Monte Carlo).** No closed form for $\mathbb E[Y_t]$.
Estimate it from $B = 10{,}000$ Milstein paths with an integration step
$h = \Delta/100$, as the manuscript does for its prediction bands. Report the
Monte Carlo standard error of $R^2$ (bootstrap over paths) to show that $B$ is
large enough. Validate the simulator on Gompertz and von Bertalanffy, where
the exact means are known.

## 3. Parameters

- **SDE parameters:** use the manuscript estimators with $\Delta = 12$ h and
  the same $K$: quadratic variation plus Girsanov for the logistic model,
  the plug-in Gaussian autoregression for Gompertz, and the closed-form
  maximum likelihood estimators for von Bertalanffy (with truncation
  $\tilde y = \min(y, 1 - 0.01)$). Validation target: reproduce the
  manuscript table (Gompertz $\hat r = 0.02889$, $\hat\sigma = 0.01437$;
  Logistic $0.04047$, $0.02656$; von Bertalanffy $0.01349$, $0.07367$;
  averages of the three trajectory-specific estimates). The original R script
  (`analisis_data.r`) is not in this repository, so the Python port must be
  checked against those numbers.
- **ODE parameters:** fit $r$ by nonlinear least squares on the original
  scale, with the same $K$ and with each run started at its own $y_0^{(j)}$.
  Fitting the ODEs separately (rather than reusing the SDE drift estimates)
  gives each deterministic model its best $R^2$, the usual reference. Also
  report the ODE evaluated at the SDE drift estimates, to separate "noise
  changes the mean" from "different estimation method".
- **Pooled versus per-run parameters:** main results use one parameter set
  per model (the averaged estimates); a sensitivity table uses per-run
  estimates.

## 4. Outputs

One table, one row per model and version:

| Model | Type | Fitted value | $\hat r$ | $\hat\sigma$ | $R^2$ (trajectory) | $R^2$ (one-step) | RMSE | $n$ |
|---|---|---|---|---|---|---|---|---|

Also report:

- $R^2$ per run ($j = 1, 2, 3$) and for the observed mean curve;
- a figure of observed versus fitted values, one panel per model, ODE and
  SDE together;
- for the SDEs, the Monte Carlo standard error (logistic only).

## 5. Implementation in this repository

1. New module `src/kefir_models/growth_sde_r2.py`:
   - reuse `read_trial_csv` and `validate_frame` from `ode_fit.py`, and
     `compute_r_squared` and `compute_rmse` from `sde_compare.py`;
   - functions `deterministic_solution(model, t, y0, r)`,
     `sde_mean(model, t, y0, r, sigma, *, rng, n_paths, h)`,
     `fit_ode_rate(model, data, K)`, `estimate_sde_parameters(model, data, K, delta)`,
     `r_squared_table(...)`;
   - command-line entry point and a JSON config
     `configs/growth_r2_config_reference.json` (K, $\Delta$, $B$, $h$,
     random seed, truncation $\epsilon$).
2. Tests (`tests/test_growth_sde_r2.py`):
   - closed-form means against Monte Carlo for Gompertz and von Bertalanffy;
   - the ODE solutions satisfy their equations (finite-difference check);
   - $R^2 = 1$ for noise-free synthetic data, and parameter recovery on
     simulated data;
   - the estimators reproduce the manuscript table within rounding.
3. Outputs to `results/growth_r2_outputs/` (CSV table plus PNG figure).
4. Documentation: a short section in `docs/07_comparision_criteria.tex` and
   `docs/results.md` describing both $R^2$ versions and the caveats below.

## 6. Caveats to state with the results

- $R^2$ is a descriptive goodness-of-fit summary. For nonlinear models it
  is not a valid model-selection criterion and can be misleading
  (Spiess and Neumeyer, BMC Pharmacology 10:6, 2010; Kvålseth, The American
  Statistician 39:279–285, 1985). Report it next to AIC, residual
  diagnostics, and interval coverage, not instead of them.
- In-sample $R^2$ is not a forecast measure. A time-based split, as in
  `07_comparision_criteria.tex` (train through $t = 132$ h, test at
  144–168 h), gives an out-of-sample version if needed.
- The choice $K = \max$ observation forces $y = 1$ and needs truncation for
  von Bertalanffy; report the sensitivity of $R^2$ to $K$ (for example, $K$
  10% above the maximum).
- The response units of the CSV remain unconfirmed (possibly g/L); $R^2$ is
  unit-free, but RMSE is not.

## 7. Open decisions

1. Trajectory $R^2$, one-step $R^2$, or both as the headline number?
2. Mean or median as the SDE fitted value?
3. Fit the ODEs independently or reuse the SDE drift estimates (or both)?
4. Keep $K = 52.1632$ or estimate $K$ jointly?
5. Include $t_0$ in $\Omega$ (45 points) to match the existing tables, or
   exclude it (42 points)?

## 8. Milestones

1. Port the three SDE estimators and reproduce the manuscript table.
2. Implement deterministic solutions, SDE means, and the ODE fits.
3. Compute both $R^2$ versions, tests passing in CI.
4. Produce the table and figure; add the documentation section.
5. Transfer the results to the manuscript (Section 6, Fitting).
