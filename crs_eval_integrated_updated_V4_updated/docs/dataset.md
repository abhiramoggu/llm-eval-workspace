# dataset.py

This module implements the **closed movie catalog**.

## Responsibilities
1. Load `opendialkg_movie_data.json` into `MOVIE_DB`.
2. Normalize and deduplicate field values (actors, directors, etc.).
3. Build helper indices for quick lookup.
4. Provide:
   - `recommend(constraints, limit)` → returns matching movies in shuffled order (randomized).
   - `get_available_genres()`
   - `get_attribute_values(field)`
   - `find_plot_keywords_in_text(text, limit)` → curated plot descriptor extraction

## Key data structures
- `MOVIE_DB[title] = {...fields...}`
- Field values are stored as lists for multi-valued fields.

## Constraints format
Constraints are a dictionary, e.g.
```python
{"genre": "Comedy", "actor": "Dev Patel"}
```
`recommend()` tries to filter the catalog using substring/containment checks.

## Limitations (must state in thesis)
- Matching is largely string-based: paraphrases and aliases may not match unless added to normalization.
