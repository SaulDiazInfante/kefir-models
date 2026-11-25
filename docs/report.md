# Report

A detailed LaTeX report accompanies the package, covering the data, the five
models, the inverse-problem formulations, and the numerical results.

The report is modular LaTeX under `docs/`. The root document is
[`00_main.tex`](https://github.com/SaulDiazInfante/kefir-models/blob/main/docs/00_main.tex),
which pulls in `setup.tex`, `abstract`, and the numbered section files
(`01_data_and_preprocessing.tex` through `14_tempered_noise.tex`) plus the
shared bibliography
[`kefir_model_comparison_references.bib`](https://github.com/SaulDiazInfante/kefir-models/blob/main/docs/kefir_model_comparison_references.bib).

## Build the PDF

Regenerate the fitted results, report tables, and figures from the repository root:

```bash
python scripts/reproduce_report.py
```

The generated numerical tables live in `docs/generated/`. The fitting outputs
include CSVs and `run_metadata.json` with effective settings and content hashes.
This command preserves the existing compiled PDF; compilation is a separate step.

```bash
cd docs
latexmk -pdf 00_main.tex
```

The report embeds the curated figures from `results/` (referenced as
`../results/...`). LaTeX auxiliary files are git-ignored; `docs/00_main.pdf`
is a tracked snapshot. Build from the sources to obtain the current manuscript.
The existing user PDF is preserved during this iteration and a separate updated
review PDF is provided.

Dedicated sections now include the controlled transition/calibration
investigation and the [Bayesian observation-model extension](bayesian-observation.md).
They retain the distinction between computational convergence, parameter
uncertainty, and predictive calibration.

The [structural discrepancy comparison](structural-discrepancy.md) crosses
shared versus independent deviations with OU versus Matérn-3/2 kernels. It
adds predictive-density and interval scores and a distinct-target one-step
comparison. Reproduce this extension after the Bayesian fits with
`python scripts/run_structural_iteration.py`.

The [independent-unit validation protocol](prospective-validation.md) adds
experimental metadata requirements, a readiness audit, and outcome-separated
forecast/scoring utilities. It is a draft awaiting new experimental inputs.
The report also derives the infinite response-mean limitation of the
unbounded noise-SD priors and specifies median point forecasts for the new protocol.

The [moment-controlled prior comparison](tempered-noise.md) adds a separate
normalized prior family with finite response moments, a global posterior
integrability bound, median matching, and 40 refits. The prospective draft
retains its earlier specification pending an explicit amendment.
