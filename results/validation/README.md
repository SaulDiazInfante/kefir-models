# Numerical validation, 2026-11-25

The CSVs were generated with `scripts/check_simulation_refinement.py` using
CPU computation and one thread. Both use the fitted parameters and Neural SDE
checkpoint of the first corrected reference run; that run's numerical results
were reproduced by the subsequent reference pipeline.

- `simulation_refinement.csv`: 3000 paths per model and setting; seeds 8, 9,
  and 10; Neural SDE maximum normalized-time steps 0.002, 0.001, and 0.0005;
  logistic maximum original-time steps 0.1, 0.05, and 0.025. Monte Carlo
  standard errors concern simulated means, not parameter uncertainty.
- `capacity_sensitivity.csv`: seed 7, 3000 epochs, shared revised
  initialization, and otherwise reference PINN settings. Each pair changes
  only the lower-bound choice and its associated softplus parameterization.

No logistic state was projected in the refinement runs. Coverage remains
about 0.50 for the Neural SDE and about 0.90 for the logistic SDE. These
finite checks do not prove asymptotic convergence or out-of-sample calibration.
See REV-07 in `docs/code-review-issues.md` for the open scientific follow-up.

Reproduce after fitting:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/check_simulation_refinement.py --capacity-sensitivity
```
