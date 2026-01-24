# LLM Scorer (PEPPER-style) Flow (context-aware/evaluate.py::llm_judge)

## Role
Provide a qualitative evaluation of the CRS using an LLM-as-judge with three human-centered criteria.

## Inputs
- Conversation transcript (USER and SYSTEM turns).
- User preference summary extracted from the conversation.
- Judge model name from `context-aware/config.py`.

## Outputs
- A dict with numeric scores: proactiveness, coherence, personalization (1-5 scale).

## Step-by-step logic
1. Build conversation history text
   - Concatenate turns as "USER: ..." and "SYSTEM: ..." lines.

2. Extract user preference summary
   - Simple regex-based extraction of preference statements (e.g., "I want", "I'm in the mood for").
   - This summary is included in the judge prompt.

3. Prompt the LLM judge
   - The prompt defines three metrics:
     - Proactiveness: initiative and guidance.
     - Coherence: logical consistency and continuity.
     - Personalization: alignment to expressed preferences.
   - The prompt requires the judge to output the three scores in a fixed format.

4. Parse judge output
   - Extract digits from each metric line.
   - Return a dict with keys: proactiveness, coherence, personalization.

## Notes to emphasize in the paper
- These are PEPPER-style subjective metrics computed by an LLM, not ground-truth labels.
- The judge uses a single prompt and a single model; results should be interpreted with caution.
