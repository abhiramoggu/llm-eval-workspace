# File: feature_extraction.py

## Purpose
Catalog-grounded concept extraction and similarity utilities for evaluation.

## Inputs
- Free-form utterance text.
- Catalog indices from `dataset.py`.
- `CONCEPT_FIELDS` from `config.py`.

## Outputs
- Concept sets and IDF-weighted attribute vectors.
- Similarity metrics (Jaccard, cosine, copy ratio).

## Key functions/classes
- `CatalogFeaturizer`
- `concept_set(text)`
- `attribute_vector(text)`
- `copy_ratio(system_text, user_text)`

## Do-not-break invariants
- Grounding must remain catalog-based (no open-ended embedding-only replacement).
- Keep `topic_set` as a legacy alias of `concept_set`.
