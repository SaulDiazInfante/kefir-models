# Checkpoint: report review and Bayesian calibration

Date: 25 November 2026. Branch: `main`. Base before this checkpoint: `67b97bb`.
The checkpoint commit is the commit containing this file.

## Saved state

The implementation, report sources, generated tables, curated figures,
experiment summaries, and project instructions cover REV-01 through REV-12.
Read `AGENTS.md` and `docs/code-review-issues.md` before continuing. Scientific
adequacy is distinct from implementation status: late underprediction,
experimental metadata, and independent-unit validation remain open.

The last iteration supplies a separate tempered noise-prior family with finite
response moments and 40 validated refits. The earlier fits and prospective
draft remain separate. All 118 tests pass; Ruff lint/format and the strict
MkDocs build pass at this checkpoint.

The tracked `docs/00_main.pdf` preserves the user's pre-review PDF. The latest
review is the separate 40-page `outputs/review_2026-11-25/report-tempered-noise.pdf`.
Review PDFs, posterior-chain NPZ files, and trained-model checkpoints remain
ignored by Git. A local recovery archive also retains those artifacts:

```text
outputs/checkpoints/checkpoint-2026-11-25T193444Z.tar.gz
outputs/checkpoints/checkpoint-2026-11-25T193444Z.manifest.json
```

These local archives are not pushed. Source and curated results in Git support
reproduction; fresh checkouts must regenerate the ignored fitted archives.
To compile the current report without replacing the tracked PDF:

```sh
cd docs
latexmk -pdf -outdir=../outputs/report-build 00_main.tex
```

## CSV formatting change pulled from upstream

The upstream pull to `67b97bb` removed a trailing empty field at time 120 in
`data/raw/waterKefirTrialsReferece.csv`. Parsed times, trial columns, and all
45 response values are exactly unchanged. The raw-file hashes are:

- Before formatting: `230236e3be3eee57ca1821a65a3bb50cb09d28d097ce5aea9746cc901fc3a1b1`.
- After formatting: `a9ca1a50f665305619db1a06be43a1dc31f16c0293499cdcc01eaf75f6261acd`.

`data/metadata/reference.json` now describes the current file and records
both hashes, Git revisions, and a numerical-identity fingerprint. The current
readiness audit was regenerated. Experimental facts remain unconfirmed.
Historical experiment and report manifests retain their original hashes;
they describe the inputs actually used at the time.

Existing strict byte-hash checks intentionally reject reuse of those older
fits against the reformatted CSV. Before using a reuse flag, implement and
validate a narrowly scoped numerical-equivalence/provenance check, or rerun
the affected experiment. Do not silently rewrite historical hashes or remove
the mismatch check. The previous exact bytes remain available from
`git show ef1f80f:data/raw/waterKefirTrialsReferece.csv`.

## Resume from here

The pull also added `docs/plans/r_squared_growth_models_plan.md`. It proposes
a separate logistic/Gompertz/von Bertalanffy and Stan-based comparison.
Reconcile its fixed-capacity and experimental assumptions with the current
report before implementing it; this checkpoint does not silently adopt them.

Confirmed units, trial/batch relationships, and additional independently
collected trajectories are still required for REV-10. The prospective
protocol is a draft and has not been frozen or externally registered.
Adopting the new tempered prior requires an explicit protocol amendment.
