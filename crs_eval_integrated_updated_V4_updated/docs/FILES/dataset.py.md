# File: dataset.py

## Purpose
Load and index the closed movie catalog; provide catalog-grounded lookups.

## Inputs
- `opendialkg_movie_data.json` (movie metadata).

## Outputs
- In-memory indices (`MOVIE_DB`, `ATTRIBUTE_INDEX`, `PLOT_KEYWORD_INDEX`).
- Retrieval helpers (`recommend`, `get_attribute_values`, `find_attribute_in_text`).

## Key functions
- `_load_movie_data()`
- `recommend(constraints, limit)`
- `find_plot_keywords_in_text(text)`

## Do-not-break invariants
- Keep schema of `MOVIE_DB` entries intact.
- Do not change normalization semantics without updating grounding docs.
