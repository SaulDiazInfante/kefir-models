# kefir-models

Neural ODE, Neural SDE, and logistic physics-informed neural network (PINN)
models for analysing **water kefir fermentation** data.

![All-model comparison](assets/img/all_models_comparison.png)

## What it does

The package fits and compares five dynamical descriptions of the same growth
experiment, on a common original response scale:

1. **Classical logistic ODE** — the Verhulst baseline (`r`, `K`).
2. **Neural ODE** — a learned right-hand side `dy/dt = f_theta(t, y)`.
3. **Neural SDE** — Itô drift + diffusion with predictive uncertainty bands.
4. **Deterministic logistic PINN** — physics-informed inverse model.
5. **Stochastic logistic PINN / SDE** — adds a diffusion parameter.

All models are scored with RMSE, R², AIC, BIC, and (for the stochastic models)
predictive-interval coverage.

## Quick links

- [Installation](installation.md)
- [Usage](usage.md) — command-line tools and the Python API
- [Models](models.md) — what each model is and how it is fitted
- [Results](results.md) — figure gallery
- [API Reference](api.md)

## At a glance

```bash
pip install -e ".[dev]"
kefir-sde-compare --config configs/sde_compare_config_reference.json
```

Source code: <https://github.com/SaulDiazInfante/kefir-models>
