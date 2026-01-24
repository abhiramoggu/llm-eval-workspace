# File: batch_run.py

## Purpose
Run multi-model simulations and evaluations, then aggregate results.

## Inputs
- `LLM_SYSTEMS` list in `config.py`.
- Simulation logs from `simulate.run_simulation`.

## Outputs
- `results.jsonl`
- `model_metrics.csv`
- Per-model judge score CSVs in `logs/`

## Key functions
- `run_batch(n_sessions)`

## Do-not-break invariants
- Do not change `results.jsonl` schema (additive only).
- Keep paired-session behavior for statistical comparisons.
