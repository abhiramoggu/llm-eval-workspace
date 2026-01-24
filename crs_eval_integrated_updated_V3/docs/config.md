# config.py

This file defines **global configuration** for simulation, evaluation, logging, and metric weights.

## Key constants

- `MODE`: `"mock"` avoids calling Ollama; `"ollama"` uses `ollama run`.
- `LLM_SYSTEMS`: list of CRS models to compare (used in `batch_run.py`).
- `USER_MODEL`, `JUDGE_MODEL`: Ollama model names for the simulator and judge.
- `TOPIC_EXTRACTOR_MODE`: `"llm"`, `"lda"`, or `"heuristic"` for legacy topic extraction.
- `N_TURNS`: number of USER turns (conversation will have `2*N_TURNS` messages).
- `LOG_DIR`: directory for saved conversation logs.
- `TAS_WEIGHTS`: `{alpha, beta, gamma}` in `TAS(t) = alpha*CC + beta*CR - gamma*I`.
- `ALIGNMENT_THRESHOLD`: CC threshold used for recovery.
- `RECOVERY_WINDOW`: how many turns after a shift we allow before declaring “recovered”.
- `EVAL_MODE`: `"grounded"` (preferred) or `"legacy"` (older experimental methods).
- `RESULTS_FILE`, `SUMMARY_CSV`, `FIG_DIR`: output paths for results and figures.
- `USE_TRANSFORMER_EMBEDDER`: toggles sentence-transformer embeddings in legacy mode.
- `CONCEPT_FIELDS`: canonical list of grounded fields (concepts).
- `TOPIC_FIELDS`: legacy alias for `CONCEPT_FIELDS` (do not remove yet).

## Notes
- Grounded concept fields are defined in `feature_extraction.FeaturizerConfig.topic_fields`.
  If you change them, update the extractor and the thesis definition of \(E(x)\).
- Terminology: “topic” is a legacy label; in this pipeline it means catalog-grounded concepts.
