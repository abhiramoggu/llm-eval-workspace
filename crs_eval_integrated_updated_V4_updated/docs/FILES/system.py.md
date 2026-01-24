# File: system.py

## Purpose
CRS wrapper that extracts constraints and generates catalog-based responses.

## Inputs
- USER messages (free-form text).
- Catalog lookups via `dataset.recommend()`.

## Outputs
- SYSTEM response text.
- Updated constraint state (`self.constraints`).
- Last recommended titles (`self.last_rec_titles`).

## Key functions/classes
- `CRSSystem.respond(user_message)`
- `_extract_constraints(user_message)`
- `_update_constraints(user_message)`

## Do-not-break invariants
- Recommendations must remain catalog-constrained.
- Keep constraint keys stable (`genre`, `actor`, `director`, `writer`, `language`, `year`, `name`, `plot`).
