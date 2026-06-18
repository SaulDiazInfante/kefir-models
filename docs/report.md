# Report

A detailed LaTeX report accompanies the package, covering the data, the five
models, the inverse-problem formulations, and the numerical results.

The report is modular LaTeX under `docs/`. The root document is
[`00_main.tex`](https://github.com/SaulDiazInfante/kefir-models/blob/main/docs/00_main.tex),
which pulls in `setup.tex`, `abstract`, and the numbered section files
(`01_data_and_preprocessing.tex` through `09_reproducibility.tex`) plus the
shared bibliography
[`kefir_model_comparison_references.bib`](https://github.com/SaulDiazInfante/kefir-models/blob/main/docs/kefir_model_comparison_references.bib).

## Build the PDF

```bash
cd docs
latexmk -pdf 00_main.tex
```

The report embeds the curated figures from `results/` (referenced as
`../results/...`). The compiled PDF and LaTeX auxiliary files are git-ignored, so
build it locally to read the full manuscript.
