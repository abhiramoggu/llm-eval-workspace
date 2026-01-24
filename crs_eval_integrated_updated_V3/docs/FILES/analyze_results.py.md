# File: analyze_results.py

## Purpose
Load `results.jsonl` and generate plots/statistics across models.

## Inputs
- `results.jsonl`
- Output directory from `config.FIG_DIR`

## Outputs
- Figures under `figures/<mode>/`
- `concept_delta_summary.json` with per-model missing/hallucinated top-k concepts

## Key functions
- `plot_radar(...)`
- `paired_model_tests(...)`
- `write_concept_delta_summary(...)`

## Do-not-break invariants
- Do not assume specific metrics exist; guard with column checks.
- Recreate figure directories on each run (current behavior).
