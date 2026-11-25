# Noise priors with finite response moments

REV-12 follows the [predictive-moment audit](prospective-validation.md#predictive-moment-audit).
It introduces and refits a separate prior family. The earlier lognormal-prior
fits, numerical summaries, and prospective draft remain separate records.
No new independent experiments or physical metadata were supplied.

## Prior and analytic guarantee

For each of the latent and measurement SDs, `v = a` or `b`, define

```text
z = log(v)
p_eta(z) = Normal(z; mu_eta, 1) * exp(-eta * exp(2*z)) / Z_eta
```

`Z_eta` is the normal expectation of the penalty, computed by one-dimensional
quadrature. The Gaussian kernel location `mu_eta` is calibrated so that the
SD median remains 0.03. It is not the mean of the tempered log-SD prior.
Both noise components receive this prior independently. The penalty is
smooth and leaves support `(0, infinity)`; it does not specify a physical
upper bound. The log-response SDs are dimensionless. Their numerical prior
medians retain the earlier exploratory choice and are not elicited evidence.

The main strength `eta=2` is fixed to give finite response means and variances.
The fixed sensitivity strength `eta=8` additionally gives finite fourth
moments. Strengths were not optimized against coverage or any other score.
The prior medians match, but the prior shapes do not:

| Strength | Gaussian kernel location | SD 5th percentile | SD median | SD 95th percentile |
|---|---:|---:|---:|---:|
| Historical 0 | -3.50656 | 0.00579 | 0.03000 | 0.15541 |
| Main 2 | -3.49260 | 0.00584 | 0.03000 | 0.14834 |
| Sensitivity 8 | -3.45864 | 0.00596 | 0.03000 | 0.13590 |

The normalized MCMC target in log coordinates includes both quadratic SD
penalties and both log normalizers. No additional log-SD Jacobian is added
to a density already specified in log coordinates. Independent quadrature
in the SD coordinate checks normalization and this transformation.

The posterior moment argument bounds the training measurement likelihood
before integrating the latent process. With N positive training responses,
that conditional likelihood is at most a data-dependent constant times
`b^(-N)`. The unconditional stationary latent discrepancy at a future time
has variance `a²`, independently of which temporal kernel or sharing
assumption is used. A future response has conditional q-th moment
`m_theta(t)^q * exp(q²*(a²+b²)/2)`, and `m_theta(t) <= K`.

After combining these facts with the prior, the posterior q-th moment is
bounded by a finite constant times the Gaussian-kernel expectation of

```text
K^q * b^(-N) * exp((q²/2 - eta) * (a² + b²)).
```

For `eta >= q²/2`, the exponential factor is at most one. The remaining
capacity positive moments and lognormal negative SD moments are finite.
The data normalizer is also positive and finite by the same likelihood
bound. This proves the result over the full joint posterior, without an
unproved bound on Gaussian conditioning weights. Controlling just measurement
SD would not give this argument for latent amplitude.

The boundary is included. For `q²/2 > eta`, the large-measurement-SD tail
argument gives divergence. Thus positive response moments exist exactly
through order `sqrt(2*eta)`, inclusive. The main prior has a finite second
moment but not a finite fourth moment. The stronger prior has a finite
fourth moment but does not make all moments finite. The report provides
the full derivation and the normalizer-ratio identity used to check
exponential-square SD moments numerically.

## Fixed experiment and results

The comparison has 40 new fits:

- Four structural models at cutoffs 108, 120, 132, 144, 156, and 168 for
  `eta=2`: 24 cases.
- The same four models at cutoffs 108, 120, 132, and 168 for `eta=8`: 16 cases.

The likelihoods, logistic priors, stationary variances, and matched e-fold
correlation-time convention remain unchanged. No old-prior posterior samples
were reused as new-prior fits. Four chains adapt only during 6000 warmup
iterations, retaining 36000 draws per chain in 38 cases and 108000 in two
cases after diagnostic extension.

All cases pass the declared computation gates. Largest rank/folded split
R-hat is 1.0059, minimum bulk ESS 928.8, and minimum tail ESS 423.4. Diagnostics
also monitor actual conditional-response-mean integrands at first/last
forecast times using every eighth retained draw. Their minimum raw ESS is
948.2 and largest estimated MCSE is 0.0717 response units. The finite second
response moment implies a finite variance for this conditional-mean
integrand by Jensen's inequality. A Markov-chain CLT still requires mixing
assumptions; finite moments and finite-chain diagnostics are not a proof.

Main-prior multi-time summaries retain the same 36 overlapping forecast
entries and 15 distinct responses used previously:

| Model | Coverage | Width | Interval score | RMSE of finite mean forecast |
|---|---:|---:|---:|---:|
| Independent OU | 0.8056 | 5.878 | 10.092 | 2.406 |
| Shared OU | 0.9722 | 9.161 | 9.449 | 2.191 |
| Independent Matérn | 0.8056 | 5.857 | 10.149 | 2.408 |
| Shared Matérn | 0.9444 | 8.827 | 9.514 | 2.246 |

Coverage is unchanged from the historical comparison, and also unchanged
under the stronger tail penalty. For 15 distinct one-step targets, coverage
remains 13/15 for the independent models and 14/15 for the shared models.
Mean-forecast bias remains positive. The repair supplies a mathematically
finite response expectation without resolving the late systematic departure.
The earlier mean-based numbers remain finite posterior-subset diagnostics.
All new tables report finite-mean and median point forecasts explicitly.

Joint density scores retain cross-trial dependence and parameter integration.
They still favor independent OU over shared OU while marginal scores favor
shared OU. Small changes across prior strengths do not establish a meaningful
ranking, especially given Monte Carlo uncertainty and reused trajectories.
No model or tail strength is promoted to the mechanistic reference.

At cutoff 132, the independent/shared OU integration check uses 6000, 12000,
and 24000 posterior components. Maximum mean shift from the 6000-component
result is 0.0136 response units; maximum interval endpoint shift is 0.0150.
Coverage is unchanged. This numerical check is separate from the analytic
proof of existence, which is not itself a useful finite-sample error bound.

## Reproduction and continuation

From the repository root with the Bayesian optional dependency installed:

```sh
python scripts/run_tempered_iteration.py
```

The runner uses four experiment processes, each with one numerical thread;
`--workers 1` makes execution sequential. `--reuse-validated` checks source
and data hashes before rescoring this family's existing samples. Posterior
archives are gitignored and regenerated by the runner. Prior normalizers,
calibrated kernel locations, source/input hashes, seeds, acceptance rates,
package versions, and numerical settings are recorded per case.

`src/kefir_models/tempered_noise.py` defines the normalized prior and target.
`scripts/investigate_tempered_noise.py` fits and scores a case.
`scripts/summarize_tempered_noise.py` writes prior checks, score/parameter
summaries, numerical refinement checks, and plots.
`scripts/build_tempered_tables.py` generates LaTeX from the saved CSVs.
The report includes `docs/14_tempered_noise.tex` and the generated tables
and interpretation. All results are under `results/tempered/`.

These new priors solve REV-11's moment issue for the new family. They do not
retroactively change the earlier priors or show that prior tail choices
have been physically justified. Noise-prior median sensitivity at half/double
the reference value has not been repeated for this family; the earlier
lognormal-family sensitivity is not a substitute. Numerical tests confirm
the new prior at those median values without constituting refitted
experimental sensitivity results.

The [prospective protocol](prospective-validation.md) retains its earlier
priors and median point forecast. Adopting the new family requires an
explicit protocol amendment, including its prior specification and source
snapshot. Confirmed experimental metadata and additional independently
collected trajectories remain unavailable; REV-07 and REV-10 stay open.
