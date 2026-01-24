# System Overview Flow (context-aware pipeline)

## Purpose
Define and evaluate a conversational recommender system (CRS) that must adapt to user preference shifts over a multi-turn dialogue, then score adaptation with both automated metrics and an LLM-as-judge.

## Core inputs
- Movie knowledge base from `context-aware/opendialkg_movie_data.json`.
- Configuration from `context-aware/config.py` (models, thresholds, weights, modes).
- Local LLMs via Ollama for CRS responses, user simulation, and LLM-as-judge.

## Core outputs
- Per-session logs: `context-aware/logs/<model>_session_<id>.json`.
- Per-session scores: `context-aware/results.jsonl`.
- Aggregated metrics: `context-aware/model_metrics.csv`.
- Figures for analysis: `context-aware/figures/...`.
- Statistical tests: `context-aware/statistical_analysis_results.txt`.

## End-to-end flow (step-by-step)
1. Dataset load and indexing
   - `context-aware/dataset.py` loads `opendialkg_movie_data.json` at import time.
   - Builds indices for genre/actor/director/language/writer/year and plot keywords.

2. Configuration and experiment setup
   - `context-aware/config.py` sets the MODE (ollama or mock), models, topic extractor mode, similarity thresholds, and CAS weights.
   - LDA model can be pre-trained using `context-aware/train_lda.py` for topic extraction.

3. Conversation simulation (per model, per session)
   - `context-aware/batch_run.py` loops through each CRS model in `LLM_SYSTEMS`.
   - For each session, `context-aware/simulate.py`:
     1) Initialize `CRSSystem(model_name)` from `context-aware/system.py`.
     2) Initialize `UserSimulator()` from `context-aware/user_sim.py`.
     3) For each of 20 turns:
        - User simulator generates a user utterance.
        - CRS updates constraints, retrieves candidate movies, and generates a reply.
        - User simulator records the CRS reply to update cues for the next turn.
     4) Save the full conversation to `context-aware/logs/`.

4. Evaluation (per session)
   - `context-aware/evaluate.py` computes:
     - Topic segmentation for user turns.
     - Cross-coherence, context retention, topic recovery, interference.
     - Context Adaptation Score (CAS) and detailed per-shift diagnostics.
   - Optional LLM-as-judge scores (proactiveness, coherence, personalization).
   - Write one JSON record per session to `context-aware/results.jsonl`.

5. Aggregation and analysis
   - `context-aware/batch_run.py` aggregates `results.jsonl` into `context-aware/model_metrics.csv`.
   - `context-aware/analyze_results.py` builds bar charts, distributions, radars, correlation heatmaps, and alignment-over-time plots under `context-aware/figures/`.
   - `context-aware/statistical_analysis.py` runs ANOVA and Tukey HSD and saves results to `context-aware/statistical_analysis_results.txt`.

## Key configuration knobs (from `context-aware/config.py`)
- `LLM_SYSTEMS`: CRS models compared in experiments (gemma:2b, qwen:7b, qwen:4b, llama3:instruct, llama2:latest, mistral:7b).
- `USER_MODEL` and `JUDGE_MODEL`: used for user simulation and LLM-as-judge (default llama3:instruct).
- `TOPIC_EXTRACTOR_MODE`: lda (default), llm, or heuristic.
- `SIM_TOPIC_SHIFT`, `TOPIC_JACCARD_SHIFT`, `ALIGNMENT_THRESHOLD`: thresholds for shift detection and recovery.
- `CAS_WEIGHTS`: weights used to combine sub-metrics into CAS.
- `USE_TRANSFORMER_EMBEDDER`: if 1, use sentence-transformer embeddings; else TF-IDF fallback.

## Validation notes (NLP reviewer check)
- Constraint extraction is heuristic, not LLM-based. The CRS LLM only crafts the final response. The write-up must reflect this accurately.
- `UserSimulator._reference_title()` returns a full movie dict, not just a title string, which creates noisy user utterances (see `context-aware/user_sim.py`). This can distort topic extraction and should be fixed or acknowledged.
- Topic segmentation detects nearly one shift per user turn (avg ~18.8 shifts per 20 turns in `context-aware/results.jsonl`), suggesting thresholds may be too sensitive or user messages too heterogeneous.
- If `USE_TRANSFORMER_EMBEDDER` is not enabled, similarity uses TF-IDF (not sentence-transformers). Any claim about transformer embeddings must match the actual run setting.
- If the LDA model files are missing, evaluation silently falls back to heuristic topic extraction, which changes the semantics of TAS. Report the topic extractor mode used in experiments.
- The OpenDialKG-derived genre list includes noisy labels (countries, entities). This affects both simulation and evaluation. Consider cleaning or describing this limitation.
- `context-aware/config.py` defines `N_TURNS`, but `context-aware/simulate.py` hard-codes 20 turns. Use one source of truth in the paper.
- LLM-as-judge uses the same family model as the user simulator by default, which can bias scores. Consider using multiple judges or note this limitation.
