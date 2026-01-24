# Pipeline Overview

This document summarizes the CRS evaluation pipeline stages and file responsibilities.

## System Diagram Narrative
The pipeline is a closed-world loop: a **User Simulator** issues grounded preference
requests, a **CRS System** responds using catalog-constrained recommendations, and
an **Evaluator** scores adaptation using grounded concept overlap and constraint
similarity. Optional LLM-judge scoring is layered on top for qualitative checks,
followed by analysis scripts that aggregate metrics across models and generate plots.

## Stages (simulate → evaluate → judge → analyze)
1. **Simulate** (`simulate.py`, `user_sim.py`, `system.py`)
   - Generates alternating USER/SYSTEM turns.
   - Attaches `user_meta` (including shift events) and system `constraints`.
   - Writes a log file in `logs/`.
2. **Evaluate** (`evaluate.py`, `feature_extraction.py`)
   - Extracts catalog-grounded concepts `E(x)` and vectors (structured fields only by default).
   - Computes CC/CR/CP per turn, TAS mean, and shift recovery metrics.
   - Computes concept delta diagnostics (missing/hallucinated concepts).
3. **Judge** (`evaluate.py`)
   - Optional LLM-as-judge rubric; results recorded under `judge`.
4. **Analyze** (`analyze_results.py`)
   - Aggregates metrics by model.
   - Produces plots and correlation summaries in `figures/`.

## File Responsibilities (core)
- `config.py`: global constants, models, thresholds, and concept field list.
- `dataset.py`: loads catalog, normalizes values, and performs constraint lookup.
- `feature_extraction.py`: grounded concept extraction and similarity functions.
- `user_sim.py`: generates preference-focused user turns with shift metadata.
- `system.py`: CRS response generation with catalog-based recommendations.
- `simulate.py`: orchestrates a single dialogue run and saves logs.
- `evaluate.py`: computes grounded/legacy metrics and judge scores.
- `batch_run.py`: runs multi-model batches and writes `results.jsonl`.
- `analyze_results.py`: loads results and produces plots/statistics.

## Legacy Naming Note
“Topic” is a legacy label. In this pipeline:
- `topic` == catalog-grounded **concept** `(field, value)`.
- `TOPIC_FIELDS` is a legacy alias for `CONCEPT_FIELDS`.
