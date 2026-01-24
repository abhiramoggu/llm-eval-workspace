# User Simulator Flow (context-aware/user_sim.py)

## Role
Generate realistic multi-turn user behavior with preference shifts (topic changes, attribute updates, and occasional contradictions).

## Inputs
- Movie dataset indices for genres and attribute values.
- User simulator settings from `context-aware/config.py` (MODE, USER_MODEL, ERROR_RATE).

## Outputs
- Next user utterance for each turn.
- Updated internal state (current topic, constraints, conversation history).

## Initialization
- Load available genres from the dataset (filtered by minimum coverage).
- Build per-genre catalogs for quick title lookup.
- Precompute attribute value pools for shifting (year, actor, director, language, writer, plot).
- Select a `true_genre` and initialize `current_topic` and `current_constraints`.
- Initialize conversation stages: `initial -> details -> awaiting_shift -> waiting_system`.

## Turn-level generation (template-driven mode)
1. Initial stage
   - Ask for recommendations in a genre.
   - With probability `ERROR_RATE`, request a wrong genre to simulate preference error.

2. Details stage
   - Ask about actors, themes, or mood based on cues in the system response.
   - Occasionally emit a contradiction (e.g., "Maybe I was wrong earlier").
   - After detail questions are exhausted, move to `awaiting_shift`.

3. Awaiting shift stage
   - Shift to a new attribute or genre using `_compose_attribute_shift_message()`.
   - Update `current_constraints` and reset detail sequence.
   - Move to `waiting_system`.

4. Waiting system stage
   - If the system responded, ask for more recommendations.
   - Otherwise, prompt the system to continue.

## Optional LLM-driven drift
- If `use_llm_drift=True`, the simulator calls an LLM to generate the next user message from conversation history with explicit rules for topic drift and contradictions.

## How system replies affect the simulator
- `record_system_message()` stores the reply and extracts:
  - Recommended titles (via string matching against known titles).
  - Cue keywords (actors/themes/mood) to drive the next question.
  - Mentioned genres to propose a potential future shift.
- This feedback loop makes the next user turn dependent on the system output.

## Notes to emphasize in the paper
- The simulator is primarily rule-based, with an optional LLM drift mode.
- Preference shifts are explicit and frequent to stress-test adaptation.
