# Reproducibility notes (minimal checklist)

This project is a simulation study with substantial underdetermination from connectome → dynamics.
The goal is to make assumptions explicit and results auditable.

## What is included
- `manuscript.pdf` and LaTeX sources (`manuscript.tex`, `references.bib`, `figures/`)
- Source code in the repository root (`src/`, `scripts/`)
- A runnable toy demo (`scripts/demo_mock_pipeline.py`) that does not require FlyWire access

## What is *not* redistributed
- FlyWire connectivity tables / matrices and any restricted data products.
  Scripts are provided to fetch them if the user has access.

## Determinism
- All assays are intended to be run across multiple random seeds.
- Seeds should be reported in outputs and stored alongside cached results.

## Environment
- Python >= 3.9
- Core deps: numpy, scipy, matplotlib, pandas, tqdm
- Optional: `fafbseg` for FlyWire access (API may change)

## Known sensitivity knobs (model degrees of freedom)
- global synapse-count scaling `alpha`
- neurotransmitter → sign mapping (esp. glutamate)
- OU noise scale and time constant
- AdEx parameters (cell-type specificity not implemented)

## What reviewers should be able to do
- Run toy demo end-to-end on a laptop
- Fetch data (with access) and rerun scripts to regenerate figures
