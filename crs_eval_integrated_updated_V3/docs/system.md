# system.py

Implements the CRS agent wrapper.

## Modes
- `MODE="mock"`: rule-based responses (good for debugging).
- `MODE="ollama"`: query an LLM with a system prompt and the catalog.

## Key responsibilities
- Parse user constraints heuristically (`extract_constraints`).
- Maintain accumulated constraints in `self.constraints`.
- Retrieve recommendations from `dataset.recommend()` in a closed world.
- Compose responses with recommendations + reasoning.

## Important for fairness
All CRS models recommend from the *same fixed catalog*, so comparisons are within a closed domain.
