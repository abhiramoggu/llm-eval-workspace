# CRS Component Flow (context-aware/system.py)

## Role
Turn each user message into a CRS response that uses current constraints and dataset-backed recommendations.

## Inputs
- User utterance (free-form text).
- Movie dataset and attribute indices from `context-aware/dataset.py`.
- Current constraint state (genre, year, actor, director, language, writer, plot).

## Outputs
- Natural language recommendation response.
- Updated constraint state (stored in `self.constraints`).

## Step-by-step logic
1. Initialize
   - `CRSSystem(model_name)` stores the CRS LLM name and an empty constraints dict.

2. Update constraints (rule-based)
   - `_extract_constraints()`:
     - Normalize genre by checking if any known genre substring appears in the user text.
     - Extract year via regex (e.g., 1999, 2008).
     - Extract attributes by matching known values in the text (actor, director, language, writer, name).
     - Extract plot keywords that overlap with dataset plot keywords.
   - `_update_constraints()`:
     - If user says "anything works" or similar, reset constraints to only genre.
     - Merge newly extracted constraints into `self.constraints`.
     - Remove any keys outside the allowed set to keep constraints compact.

3. Retrieve recommendations
   - Call `dataset.recommend(self.constraints)`.
   - Intersect candidate sets for multiple constraints; fall back to union if empty.
   - Limit to top 4 movies for prompt brevity.

4. Construct LLM prompt
   - Build a summary string of constraints.
   - Provide a short list of candidate titles and short metadata.
   - Instruct the LLM to recommend 1-2 items from the list, in 1-3 sentences.

5. Generate response
   - Call `ollama run <model_name>` with the constructed prompt.
   - Return the generated response text.

## Notes to emphasize in the paper
- The CRS uses rule-based constraint extraction from user text.
- The LLM is used for natural language generation, not for parsing constraints.
- Recommendation retrieval is dataset-backed and deterministic given constraints.
