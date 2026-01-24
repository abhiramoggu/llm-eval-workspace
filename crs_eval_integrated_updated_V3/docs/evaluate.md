# evaluate.py

Computes metrics from a saved conversation log.

## Two modes
- `EVAL_MODE="grounded"` (default):
  - uses `feature_extraction.DEFAULT_FEATURIZER`
  - computes CC/CR/I/TAS turn-level
  - uses simulator `shift_event` metadata for segment boundaries + recovery

- `EVAL_MODE="legacy"`:
  - older experimental methods (heuristics/LDA/LLM topic extraction)
  - preserved for backwards compatibility

## Outputs (grounded)
Returns a dict with:
- `cross_coherence` = mean CC
- `context_retention` = mean CR
- `topic_interference` = mean I
- `context_adaptation_score` = mean TAS
- `topic_recovery_rate`, `avg_recovery_delay`
- `missing_concepts_total`, `hallucinated_concepts_total` (aggregate concept deltas)
- `detail`: per-turn arrays, segments, shift points

These fields are written to `results.jsonl` by `batch_run.py`.

## Notes
- In grounded mode, per-turn TAS is clipped to \([-1, 1]\) for stability before averaging.
- If simulator shift metadata is present (`user_meta` or `shift_events`), it is the source of truth; otherwise, heuristic segmentation is used.
- Terminology: “topic” is a legacy label for catalog-grounded concepts; `TOPIC_FIELDS` is an alias of `CONCEPT_FIELDS`.
