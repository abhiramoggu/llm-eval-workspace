You are writing four thesis sections based on the project notes and flow files in `thesis_context/`:
- Proposed Approach
- Dataset
- Experiments
- Results and Discussion

Use the following sources as the single source of truth:
- `thesis_context/system_overview_flow.md`
- `thesis_context/crs_component_flow.md`
- `thesis_context/user_simulator_flow.md`
- `thesis_context/llm_scorer_pepper_flow.md`
- `thesis_context/tas_metric_flow.md`
- `thesis_context/experiments_results_notes.md`

Writing requirements:
1) Proposed Approach
   - Start with a system overview paragraph describing the end-to-end pipeline.
   - Add subsections for: CRS component, User simulator, LLM scorer (PEPPER-style), TAS/CAS metric.
   - Include an IEEE-style algorithm listing for the full pipeline (input -> output steps).
   - When describing the CRS, state that constraint extraction is rule-based and LLM is used for response generation.
   - When describing TAS/CAS, provide the exact formulas (cross-coherence, context retention, recovery rate, interference, weighted aggregation).

2) Dataset
   - Make it explicit that no new dataset is created; the system uses `opendialkg_movie_data.json` as a knowledge base.
   - Describe preprocessing: normalization of genres, attribute indexing, plot keyword extraction.
   - Describe how the dataset is used: recommendation retrieval, user simulation attribute pools, LDA training.
   - Include checks and fallbacks (e.g., fallback to heuristic topic extraction if LDA files are missing).
   - Mention limitations (noisy genre labels, duplicates) and how that affects evaluation.

3) Experiments
   - Describe model list, number of sessions, turns per session, and fixed settings (user model, judge model).
   - State the topic extractor mode and embedding backend used in the reported results.
   - List all metrics reported (TAS/CAS + PEPPER-style) and explain what each captures.
   - Mention statistical analysis (ANOVA + Tukey HSD).

4) Results and Discussion
   - Summarize which models lead on each metric (use values from `context-aware/model_metrics.csv`).
   - Discuss tradeoffs between recovery speed, coherence, and interference.
   - Tie results to figures (bar charts, distributions, radar charts, correlation heatmap, alignment over time).
   - Include limitations and threats to validity (LLM judge bias, topic extraction sensitivity, user simulator noise).

Style:
- Academic tone, concise and formal.
- Use consistent terminology: refer to CAS in code and TAS in the thesis (explain the mapping).
- Do not invent additional methods or datasets.
- Use ASCII only.
