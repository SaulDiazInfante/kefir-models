# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-17
### Added
- Initial public release packaged from the SPPS UNAM numerical experiments.
- `kefir_models` package: classical logistic ODE baseline, Neural ODE
  (`ode_fit`), Neural SDE comparison (`sde_compare`), logistic PINN inverse
  models (`logistic_pinn_compare`), shared plotting style (`plot_style`), and
  plotting CLIs (`plot_ode_fit`, `plot_pinn_fit`, `plot_sde_compare`,
  `plot_all_models`).
- Console-script entry points for all CLIs.
- pytest suite covering CLI parsers, data validation, scaling, metrics, and
  plotting helpers.
- MIT license, packaging metadata, ruff/pytest/coverage configuration,
  pre-commit hooks, and GitHub Actions CI.
- LaTeX comparison report sources and curated experiment figures.

[Unreleased]: https://github.com/SaulDiazInfante/kefir-models/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SaulDiazInfante/kefir-models/releases/tag/v0.1.0
