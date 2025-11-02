# Project Context

## Purpose
- Evaluate conversational recommender systems (CRS) on their ability to recover from user errors over multi-turn dialogues.
- Pair a CRS assistant with an LLM-driven user simulator, log the conversation, and score it with rule-based metrics and an LLM judge.

## Core Scripts
- `simulate.py`: Runs a single CRS↔user session (default 3 turns), prints dialogue, and writes `logs/session_<id>.json`. When executed directly it also triggers both evaluation layers.
- `batch_run.py`: Orchestrates many simulations (`n_sessions`, default 500) across `config.LLM_SYSTEMS`, persisting per-model JSON and an aggregated CSV in `results/`.
- `system.py`: Minimal CRS implementation. Extracts the user's desired genre via `call_ollama`, fetches recommendations from `dataset.recommend`, then prompts the model to craft a reply.
- `user_sim.py`: Simulated user with stochastic genre mistakes (controlled by `config.ERROR_RATE`). Later turns can call Ollama for natural responses; note the duplicated `fuser_prompt` assignment around line 51 that could be cleaned up.
- `evaluate.py`: Supplies rule-based recovery metrics and an Ollama-backed judge (`llm_judge`).

## Configuration (`config.py`)
- `MODE`: `"ollama"` to hit local models, `"mock"` for deterministic strings (useful when Ollama is unavailable).
- `LLM_SYSTEMS`: List of CRS model names to test in batch runs.
- `USER_MODEL`, `JUDGE_MODEL`: Ollama model identifiers for user simulation and evaluation.
- `ERROR_RATE`: Probability that the user states the wrong genre on turn 0.

## Data & Outputs
- `dataset.py`: Hard-coded movie list with genres; `recommend()` returns the top two titles matching the extracted genre.
- `logs/`: Session transcripts written as JSON by `simulate.run_simulation`.
- `results/`: Batch summaries (`<model>_results.json`, `batch_results.csv`) created by `batch_run.py`.

## Utilities
- `model_check.py`: Smoke-test multiple Ollama models; returns first-line responses or errors/timeouts.
- `ollama_test.py`: Quick single-model prompt helper.

## Tips for Future Work
- Switch to `MODE="mock"` during development to avoid Ollama calls.
- Extend `dataset.FAKE_MOVIES` and adjust `recommend()` when testing richer recommendation logic.
- Add defensive parsing in `system.call_ollama` and `user_sim.call_ollama` for malformed JSON or streaming responses.
- Consider logging turn-level diagnostics (e.g., extracted constraints) to ease evaluation debugging.
