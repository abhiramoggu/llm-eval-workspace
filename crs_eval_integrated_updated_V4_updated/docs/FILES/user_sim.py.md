# File: user_sim.py

## Purpose
Simulate user behavior with grounded preference focus and controlled shifts.

## Inputs
- Catalog value pools from `dataset.py`.
- Random seed for reproducibility.

## Outputs
- USER utterances.
- `user_meta` with focus fields, constraints, and shift metadata.

## Key functions/classes
- `UserSimulator.get_message()`
- `UserSimulator.get_last_meta()`
- `ShiftEvent` dataclass

## Do-not-break invariants
- Maintain `user_meta.shift_event` and `shift_event_obj` for evaluation.
- Keep grounded bridge concepts in shift policy.
