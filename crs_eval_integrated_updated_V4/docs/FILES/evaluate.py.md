# File: evaluate.py

## Purpose
Compute grounded and legacy evaluation metrics, plus optional LLM-judge scores.

## Inputs
- Conversation list (or full log dict) with USER/SYSTEM turns.
- Optional `true_genre` and `shift_events`.

## Outputs
- Metrics dict (used by `batch_run.py` / `run_evaluations.py`).
- `detail` with per-turn and segment diagnostics.

## Key functions
- `evaluate(conversation, true_genre=None, mode=None)`
- `evaluate_grounded(...)`
- `evaluate_legacy(...)`
- `llm_judge(conversation)`

## Do-not-break invariants
- Preserve existing metric keys; only add aliases.
- Prefer simulator shift metadata when present; fall back to heuristics otherwise.
