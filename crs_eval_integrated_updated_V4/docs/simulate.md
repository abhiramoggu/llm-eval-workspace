# simulate.py

Runs one conversation:
USER → SYSTEM → USER → ... for `N_TURNS`.

## Output log schema
Saved to `logs/<model>_session_<id>.json`:

- `session_id`
- `model_name`
- `true_genre`
- `conversation`: list of alternating messages:
  - USER: `{speaker, text, user_meta}`
  - SYSTEM: `{speaker, text, constraints}`
- `shift_events`: simulator shift-event list (for debugging)

This log is the input to `batch_run.py` / `evaluate.py`.
