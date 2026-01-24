# File: config.py

## Purpose
Central configuration for simulation, evaluation, and output paths.

## Inputs
- Environment variables (e.g., `SIM_MODE`, `EVAL_MODE`, `USE_TRANSFORMER_EMBEDDER`).
- Catalog-derived genres via `dataset.get_available_genres()`.

## Outputs
- Global constants used by other modules (models, thresholds, weights, paths).

## Key definitions
- `CONCEPT_FIELDS` (canonical grounded fields).
- `TOPIC_FIELDS` (legacy alias).
- `TAS_WEIGHTS`, `RECOVERY_WINDOW`, `ALIGNMENT_THRESHOLD`.

## Do-not-break invariants
- Keep `TOPIC_FIELDS = CONCEPT_FIELDS` (legacy compatibility).
- Do not remove existing constants; only add new ones or aliases.
