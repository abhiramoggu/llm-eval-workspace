# feature_extraction.py

Implements **catalog-grounded feature extraction** and similarity functions.

Terminology note: “topic” is a legacy label in code. In this pipeline it means
catalog-grounded **concepts** `(field=value)`, and `TOPIC_FIELDS` is an alias of
`CONCEPT_FIELDS`.

## Core idea
Given an utterance `x`, extract:
- a **set** of grounded concepts `E(x)` for Jaccard (CC)
- a **TF–IDF-like vector** `phi(x)` for cosine (CR)

By default, TAS uses **structured fields only** (`genre, actor, director, writer, language, year`).
Title mentions (`name`) and plot keywords (`plot_kw`) are excluded unless explicitly enabled.

## Key components
- `CatalogFeaturizer`:
  - `topic_set(text)` → returns set of strings like `"actor=Dev Patel"`.
  - `attribute_vector(text)` → returns IDF-weighted vector over the same vocabulary.
  - Optional LLM-based extraction can be enabled with `CONCEPT_EXTRACTOR_MODE=llm`.

- Similarities (canonical names; legacy aliases in parentheses):
  - `jaccard(set_a, set_b)` (`jaccard_similarity`)
  - `cosine(vec_a, vec_b)` (`cosine_similarity`)
  - `copy_ratio(system_text, user_text)` (`copying_ratio`, `ngram_copy_ratio`) → n-gram overlap penalty (I)

## Why both Jaccard and Cosine?
- Jaccard: discrete concept overlap; good for boundary/shift detection.
- Cosine: weighted constraint coherence; good when multiple constraints accumulate.
