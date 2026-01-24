# Schemas

This file defines the **authoritative schemas** for conversation objects, log files, and `results.jsonl`.
All changes must be **additive** (no renames/removals) unless all downstream readers are updated.

## Conversation (in-memory)
List of turn dicts. Required keys by speaker:

USER turn:
```json
{"speaker": "USER", "text": "...", "user_meta": {...}}
```

SYSTEM turn:
```json
{"speaker": "SYSTEM", "text": "...", "constraints": {...}}
```

Notes:
- `user_meta` is optional in the global contract but always present in simulator logs.
- `constraints` is required for SYSTEM turns.

## Log file (`logs/<model>_session_<id>.json`)
```json
{
  "session_id": "<id>",
  "model_name": "<model>",
  "true_genre": "<user.true_genre>",
  "conversation": [ ...turn dicts... ],
  "shift_events": [ ... ]
}
```

Notes:
- `shift_events` is additive metadata; evaluation prefers it when present.
- `conversation` must alternate USER/SYSTEM turns.

## Results file (`results.jsonl`)
Each line is a JSON object:
```json
{
  "model": "<model>",
  "...": "metrics from evaluate.evaluate(...)",
  "judge": { "...": "from evaluate.llm_judge(...)" }
}
```

### Required metrics (core outputs)
The following keys must exist (legacy keys remain unchanged; aliases are added):

- `concept_overlap_mean` (alias of `cross_coherence`)
- `weighted_constraint_similarity_mean` (alias of `constraint_similarity_mean` / `context_retention`)
- `copying_penalty_mean` (alias of `topic_interference`)
- `trajectory_adaptation_score_mean` (alias of `adaptation_score_mean` / `context_adaptation_score`)
- `concept_recovery_rate` (alias of `topic_recovery_rate`)
- `avg_recovery_delay`
- `missing_concepts_total`
- `hallucinated_concepts_total`

### Concept delta diagnostics (additive)
Per-conversation concept delta fields:
- `missing_concepts_per_turn` (list of lists)
- `hallucinated_concepts_per_turn` (list of lists)
- `missing_concepts_topk` (list of `{concept,count}`)
- `hallucinated_concepts_topk` (list of `{concept,count}`)

### Legacy naming note
`TOPIC_FIELDS` is a legacy alias for `CONCEPT_FIELDS`. In docs, “topic” means
catalog-grounded concept pairs `(field,value)`, not LDA topics.
