# analyze_results.py

Loads `results.jsonl` and produces:

## 1. Aggregate tables
- mean/variance of TAS, CC, CR, I, recovery rate by model
- count of missing judge scores, etc.

## 2. Correlation / validity checks
- Spearman correlation between TAS and LLM-as-judge score
- bootstrap confidence intervals
- optional Kendall tau if needed

## 3. Paired model comparisons
If `session_id` is shared across models:
- Wilcoxon signed-rank tests (or paired t-test fallback)

## 4. Visualizations
- TAS vs Judge scatter with correlation stats
- Example TAS timeline with shift markers
- (existing plots) box/violin-style summaries
- Per-field missing/hallucinated concept CSVs and debug samples (`concept_debug_samples.jsonl`)

All plots saved under `figures/`.

## Important behavior
- The script deletes and recreates the figure output directory on each run.

## Typical usage
```bash
python analyze_results.py
```
