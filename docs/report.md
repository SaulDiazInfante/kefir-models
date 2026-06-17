# Report

A detailed LaTeX report accompanies the package, covering the data, the five
models, the inverse-problem formulations, and the numerical results.

The sources live in the repository under `docs/`:

- [`kefir_model_comparison_report.tex`](https://github.com/SaulDiazInfante/kefir-models/blob/main/docs/kefir_model_comparison_report.tex)
- [`kefir_model_comparison_references.bib`](https://github.com/SaulDiazInfante/kefir-models/blob/main/docs/kefir_model_comparison_references.bib)

## Build the PDF

```bash
cd docs
latexmk -pdf kefir_model_comparison_report.tex
```

The report embeds the curated figures from `results/` (referenced as
`../results/...`). The compiled PDF and LaTeX auxiliary files are git-ignored, so
build it locally to read the full manuscript.
