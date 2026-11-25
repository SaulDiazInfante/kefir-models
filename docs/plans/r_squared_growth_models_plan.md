# Plan: deterministic versus stochastic growth models — $R^2$ and Bayesian estimation

Status: plan, revised 25 November 2026. No code has been written yet.

## Goal

Compare the deterministic (ODE) and stochastic (SDE) versions of three growth
laws on the water kefir data (`data/raw/waterKefirTrialsReferece.csv`), to
evaluate both their **fit performance** and their **parameter estimation**:

| Family | Deterministic (ODE) | Stochastic (SDE, multiplicative noise) |
|---|---|---|
| Logistic | $\dot y = r\,y(1-y)$ | $dY_t = r\,Y_t(1-Y_t)\,dt + \sigma Y_t\,dW_t$ |
| Gompertz | $\dot y = -r\,y\log y$ | $dY_t = -r\,Y_t\log Y_t\,dt + \sigma Y_t\,dW_t$ |
| von Bertalanffy | $\dot y = r(1-y)$ | $dY_t = r(1-Y_t)\,dt + \sigma(1-Y_t)\,dW_t$ |

Here $y = x/K$ with the fixed value $K = \max_{i,j} x_i^{(j)} = 52.1632$
used in the manuscript, and results are reported on the original scale
$x = K y$. The data contain three independent flask runs $j = 1, 2, 3$,
observed at $t_i = 12\,i$ h, $i = 0, \dots, 14$ ($\Delta = 12$ h), so
$n = 45$ observations.

Part A computes $R^2$ with frequentist estimates. Part B proposes a Bayesian
estimation framework that yields posterior $R^2$ distributions and a
principled comparison.

## Decisions taken (25 November 2026)

1. Report **both** the trajectory $R^2$ and the one-step-ahead $R^2$.
2. Use the **median** as the SDE fitted value.
3. Fit the ODEs **on their own** (independent least squares).
4. **Keep** $K = 52.1632$ fixed.
5. Use all **45** observations, including $t_0$, for the trajectory $R^2$,
   matching the existing comparison tables.

---

## Part A — $R^2$ with point estimates

### A1. Definition

Consistent with `docs/07_comparision_criteria.tex` and `compute_r_squared` in
`src/kefir_models/sde_compare.py`:

$$
R^2 = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}, \qquad
\mathrm{SSE} = \sum_{(i,j)\in\Omega} \bigl(x_i^{(j)} - \widehat x_i^{(j)}\bigr)^2, \qquad
\mathrm{SST} = \sum_{(i,j)\in\Omega} \bigl(x_i^{(j)} - \bar x\bigr)^2 .
$$

Only the fitted value $\widehat x_i^{(j)}$ depends on the model.

### A2. Fitted values

**Trajectory $R^2$** ($|\Omega| = 45$). Start each run at its first
observation and use the median of $Y_{t_i}$ given $Y_0 = y_0^{(j)}$ for the
SDE, and the solution $y(t_i; y_0^{(j)})$ for the ODE. At $t_0$ the residual
is zero by construction, as in the existing tables.

**One-step-ahead $R^2$** ($|\Omega| = 42$, since $t_0$ has no predecessor).
Use the median of $Y_{t_i}$ given $Y_{t_{i-1}} = y_{i-1}^{(j)}$ for the SDE,
and the ODE flow over one interval $\Delta$ from $y_{i-1}^{(j)}$.

Both versions use the same formulas below, with $(t, y_0)$ replaced by
$(t_i, y_0^{(j)})$ or by $(\Delta, y_{i-1}^{(j)})$. Write $\phi_t = e^{-rt}$.

**ODE solutions.**

$$
y_{\mathrm L}(t) = \frac{y_0 e^{rt}}{1 - y_0 + y_0 e^{rt}}, \qquad
y_{\mathrm G}(t) = \exp\bigl(\phi_t \log y_0\bigr), \qquad
y_{\mathrm V}(t) = 1 - (1 - y_0)\,\phi_t .
$$

**SDE medians.** Since the medians of monotone transforms are transforms of
medians, Gompertz and von Bertalanffy have closed forms.

- Gompertz: $Z_t = \log Y_t$ is an Ornstein–Uhlenbeck process with mean
  $m(t) = \phi_t \log y_0 - \frac{\sigma^2}{2r}(1 - \phi_t)$, so
  $\operatorname{med}(Y_t) = \exp\bigl(m(t)\bigr)$.
- von Bertalanffy: $\log(1 - Y_t)$ is a Brownian motion with drift
  $-(r + \sigma^2/2)$, so
  $\operatorname{med}(Y_t) = 1 - (1 - y_0)\,e^{-(r + \sigma^2/2)t}$.
- Logistic: no closed form; estimate the median from $B = 10{,}000$ Milstein
  paths with step $h = \Delta/100$, and report the Monte Carlo standard error
  of $R^2$ (bootstrap over paths). Validate the simulator against the
  closed-form Gompertz and von Bertalanffy medians.

Note that the SDE medians differ from the ODE solutions through the
$\sigma^2$ terms, even at equal $r$.

### A3. Parameters

- **SDEs:** the manuscript estimators with $\Delta = 12$ h — quadratic
  variation plus Girsanov (logistic), plug-in Gaussian autoregression
  (Gompertz), closed-form maximum likelihood (von Bertalanffy, with
  truncation $\tilde y = \min(y, 1 - 0.01)$). Validation target, the
  manuscript table (averages of the three run-specific estimates): Gompertz
  $\hat r = 0.02889$, $\hat\sigma = 0.01437$; logistic $0.04047$, $0.02656$;
  von Bertalanffy $0.01349$, $0.07367$. The original R script
  (`analisis_data.r`) is not in this repository, so the Python port must
  reproduce these numbers.
- **ODEs:** $r$ fitted independently by nonlinear least squares on the
  original scale, with the same $K$ and each run started at $y_0^{(j)}$.
- Main results use one parameter set per model; a sensitivity table uses the
  run-specific estimates.

### A4. Comparison outputs

| Model | Version | $\hat r$ | $\hat\sigma$ | $R^2$ trajectory | $R^2$ one-step | RMSE |
|---|---|---|---|---|---|---|
| Logistic | ODE / SDE | | | | | |
| Gompertz | ODE / SDE | | | | | |
| von Bertalanffy | ODE / SDE | | | | | |

Plus: $R^2$ per run and for the observed mean curve; a figure of observed
versus fitted values (ODE and SDE per panel); a table of $\hat r$ (ODE)
versus $\hat r$ (SDE) per model, which addresses the *estimation* side of the
comparison.

---

## Part B — Bayesian estimation framework

### B1. Why Bayesian here

- With 3 runs and 15 points each, uncertainty in $(r, \sigma)$ is large;
  posteriors quantify it directly, and posterior predictive checks compare
  the ODE and SDE versions on equal footing.
- A hierarchical model shares information across the three runs instead of
  averaging run-specific estimates.
- The framework separates **process noise** (the SDE diffusion) from
  **measurement error**, which the current estimators cannot do.
- $R^2$ becomes a posterior distribution ("Bayesian $R^2$"; Gelman, Goodrich,
  Gabry and Vehtari, *The American Statistician*, 2019), so Part A's numbers
  gain credible intervals.

### B2. Likelihoods

**Deterministic models.** Observation model on the original scale,
$x_i^{(j)} = K\,y(t_i; y_0^{(j)}, r) + \varepsilon_i^{(j)}$, with Gaussian
error (or lognormal error on $\log x$); parameters $(r, \tau)$, where $\tau$
is the error standard deviation.

**Stochastic models, no measurement error.**

- Gompertz: exact Gaussian transitions of $Z = \log Y$ (the
  Ornstein–Uhlenbeck mean and variance above) give an exact likelihood.
- von Bertalanffy: exact Gaussian increments of $\log(1 - Y)$; exact
  likelihood (requires $y < 1$, with the same truncation).
- Logistic: no closed-form transition. Options, in increasing cost:
  (i) Euler–Maruyama likelihood with $M$ latent intermediate points per
  interval (data augmentation, non-centered parameterization, sampled with
  Hamiltonian Monte Carlo); (ii) linear noise approximation with a Kalman
  filter, as in Heydari et al. (2014) for stochastic logistic growth;
  (iii) particle marginal Metropolis–Hastings with a modified diffusion
  bridge (Golightly and Wilkinson; Whitaker et al., 2017).

**Stochastic models with measurement error (state-space form).**
Latent SDE path plus $x_i^{(j)} = K\,Y_{t_i}^{(j)} \cdot e^{\eta_i^{(j)}}$,
$\eta \sim \mathcal N(0, \tau^2)$. For Gompertz this is a linear Gaussian
state-space model in $\log$ scale, so the Kalman filter gives the exact
marginal likelihood; for the logistic and von Bertalanffy models use
augmentation or particle MCMC. Whether $\sigma$ and $\tau$ are identifiable
with 45 points must be checked by simulation (B5).

### B3. Hierarchical structure (three runs)

$$
\log r_j \sim \mathcal N(\mu_r, s_r^2), \qquad
\log \sigma_j \sim \mathcal N(\mu_\sigma, s_\sigma^2), \qquad j = 1, 2, 3,
$$

with weakly informative priors on the hyperparameters. With only three runs,
$s_r$ and $s_\sigma$ need informative half-normal priors; the pooled model
(one $r$, one $\sigma$) serves as the baseline.

### B4. Priors

- $r$ (1/h): weakly informative lognormal centered on the frequentist
  estimates' order of magnitude ($\approx 0.01$–$0.05$), or on literature
  values for water kefir grain growth (to be compiled).
- $\sigma$ (1/h$^{1/2}$) and $\tau$: half-normal or exponential priors.
- Prior predictive checks: simulated trajectories must stay within plausible
  biomass ranges.

### B5. Validation workflow

1. Prior predictive checks.
2. Simulation-based calibration and parameter recovery on synthetic data with
   the observed design (3 runs, 15 points, $\Delta = 12$ h).
3. Fit to the data with HMC (Stan via CmdStanPy, or NumPyro); diagnostics
   $\hat R < 1.01$, effective sample sizes, no divergences.
4. Posterior predictive checks: trajectories, prediction-interval coverage,
   residual Q–Q plots.

### B6. Comparison criteria

- Posterior $R^2$ (trajectory and one-step, median fitted values) for every
  ODE and SDE model.
- Expected log predictive density: PSIS-LOO for the deterministic models;
  **leave-future-out** cross-validation for the time series (Bürkner, Gabry
  and Vehtari, 2020), using the one-step predictive densities.
- Posterior means and 95% credible intervals of $r$ (ODE versus SDE) and
  $\sigma$; agreement with the frequentist estimates of Part A.
- Interval coverage of the posterior predictive bands.

### B7. Implementation

- New subpackage `src/kefir_models/bayes/` with Stan programs
  (`ode_growth.stan`, `gompertz_sde.stan`, `vonbertalanffy_sde.stan`,
  `logistic_sde_em.stan`) and a Python driver; configs under `configs/`;
  outputs under `results/bayes_outputs/`.
- Tests: recovery on synthetic data; agreement of the exact Gompertz and
  von Bertalanffy likelihoods with the frequentist estimators when priors
  are flat.

---

## Caveats

- $R^2$ is descriptive; for nonlinear models it is not a valid
  model-selection criterion (Spiess and Neumeyer, *BMC Pharmacology* 10:6,
  2010; Kvålseth, *The American Statistician* 39:279–285, 1985). Use it next
  to predictive criteria, residual diagnostics, and coverage.
- In-sample fit is not forecast skill; leave-future-out or a time split
  (train through $t = 132$ h, test at 144–168 h) addresses that.
- $K = \max$ observation forces $y = 1$ and needs truncation for von
  Bertalanffy; report sensitivity to $K$.
- CSV response units remain unconfirmed (possibly g/L); $R^2$ is unit-free,
  RMSE is not.

## Milestones

1. Port the three SDE estimators; reproduce the manuscript table.
2. Part A: ODE fits, SDE medians, both $R^2$ versions, tests in CI.
3. Part B: Gompertz and von Bertalanffy Bayesian fits (exact likelihoods),
   then deterministic models, then the logistic SDE.
4. Measurement-error and hierarchical extensions, after identifiability
   checks.
5. Comparison table and figures; transfer results to the manuscript.

## References

- Bürkner, P.-C., Gabry, J., Vehtari, A. (2020). Approximate leave-future-out
  cross-validation for Bayesian time series models. *Journal of Statistical
  Computation and Simulation* 90(14).
- Gelman, A., Goodrich, B., Gabry, J., Vehtari, A. (2019). R-squared for
  Bayesian regression models. *The American Statistician* 73(3).
- Golightly, A., Wilkinson, D. J. (2008). Bayesian inference for nonlinear
  multivariate diffusion models observed with error. *Computational
  Statistics & Data Analysis* 52(3).
- Heydari, J., Lawless, C., Lydall, D. A., Wilkinson, D. J. (2014). Fast
  Bayesian parameter estimation for stochastic logistic growth models.
  *Biosystems* 122. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4169184/>
- Whitaker, G. A., Golightly, A., Boys, R. J., Sherlock, C. (2017). Bayesian
  inference for diffusion-driven mixed-effects models. *Bayesian Analysis*
  12(2). <https://arxiv.org/abs/1507.06807>
- Whitaker, G. A., Golightly, A., Boys, R. J., Sherlock, C. (2017). Improved
  bridge constructs for stochastic differential equations. *Statistics and
  Computing* 27. <https://arxiv.org/abs/1509.09120>
- Spiess, A.-N., Neumeyer, N. (2010). An evaluation of $R^2$ as an inadequate
  measure for nonlinear models in pharmacological and biochemical research.
  *BMC Pharmacology* 10:6.
- Kvålseth, T. O. (1985). Cautionary note about $R^2$. *The American
  Statistician* 39(4).

Bibliographic details marked from memory (volume/issue numbers) must be
verified before citing in the manuscript.
