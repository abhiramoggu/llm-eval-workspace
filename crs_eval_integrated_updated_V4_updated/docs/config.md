# config.py

This file defines **global configuration** for simulation, evaluation, logging, and metric weights.

## Key constants

- `MODE`: `"mock"` avoids calling Ollama; `"ollama"` uses `ollama run`.
- `LLM_SYSTEMS`: list of CRS models to compare (used in `batch_run.py`).
- `USER_MODEL`, `JUDGE_MODEL`: Ollama model names for the simulator and judge.
- `TOPIC_EXTRACTOR_MODE`: `"llm"`, `"lda"`, or `"heuristic"` for legacy topic extraction.
- `CONCEPT_EXTRACTOR_MODEL`: model used when `CONCEPT_EXTRACTOR_MODE="llm"`.
- `CONCEPT_EXTRACTOR_MODE`: `"catalog"` (default), `"embed"`, or `"llm"` for grounded concept extraction.
- `CONCEPT_EMBED_THRESHOLD`: minimum similarity for `"embed"` mode (char n-gram TF-IDF).
- `CONCEPT_EMBED_TOP_K`: top-K candidates per field for `"embed"` mode.
- `N_TURNS`: number of USER turns (conversation will have `2*N_TURNS` messages).
- `N_SESSIONS`: number of conversations per model in batch runs.
- `SHIFT_AFTER_TURNS`: force the first simulator shift after this USER turn.
- `LOG_DIR`: directory for saved conversation logs.
- `TAS_WEIGHTS`: `{alpha, beta, gamma}` in `TAS(t) = alpha*CC + beta*CR - gamma*I`.
- `ALIGNMENT_THRESHOLD`: CC threshold used for recovery.
- `RECOVERY_WINDOW`: how many turns after a shift we allow before declaring “recovered”.
- `EVAL_MODE`: `"grounded"` (preferred) or `"legacy"` (older experimental methods).
- `RESULTS_FILE`, `SUMMARY_CSV`, `FIG_DIR`: output paths for results and figures.
- `USE_TRANSFORMER_EMBEDDER`: toggles sentence-transformer embeddings in legacy mode.
- `ENABLE_REC_SAT`: enables recommendation satisfaction proxy (disabled by default).
- `STRUCTURED_CONCEPT_FIELDS`: canonical grounded fields for TAS/diagnostics.
- `CONCEPT_FIELDS`: alias for `STRUCTURED_CONCEPT_FIELDS`.
- `TOPIC_FIELDS`: legacy alias for `CONCEPT_FIELDS` (do not remove yet).
- `TITLE_FIELD`: `"name"` (excluded from TAS; used for recommendation proxy).
- `PLOT_FIELD`: `"plot_kw"` (excluded from TAS by default).
- `USE_PLOT_KW_FOR_TAS`, `USE_NAME_FOR_TAS`: feature flags to include extra fields in TAS.

## Notes
- Grounded concept fields are defined in `feature_extraction.FeaturizerConfig.topic_fields`.
  By default, TAS uses structured fields only (no `name` or `plot_kw`).
  If you change them, update the extractor and the thesis definition of \(E(x)\).
- Terminology: “topic” is a legacy label; in this pipeline it means catalog-grounded concepts.
