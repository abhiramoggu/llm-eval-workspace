# user_sim.py

User Simulator with **dynamic preference focus** and **controlled focus shifts**.

## High-level behavior
- Starts genre-centric: asks for recommendations in an initial genre.
- Asks detail questions about an **anchor title** (actors/themes/mood).
- Occasionally issues a **shift event**:
  - chooses an overlapping **bridge concept** from the current anchor context
  - switches focus field (e.g., genre → actor)
  - updates constraints and selects a new anchor consistent with constraints

## Metadata output
Every USER message has `user_meta` logged by `simulate.py`:
- `focus_field`, `focus_value`
- `constraints` (dict)
- `anchor_title`
- `shift_event` boolean
- `shift_event_obj` (from/to focus, bridge field/value, anchors)

This metadata is used in `evaluate.py` for segment boundaries + recovery metrics.

## Determinism
`seed` makes the simulator policy deterministic (LLM outputs may still vary).

## Limitations
- The simulator is a policy, not a human: it constrains shift style.
- Must report policy and shift distribution in experiments.
