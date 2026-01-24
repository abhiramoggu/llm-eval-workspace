# Experiments and Results Notes (context-aware)

## Experimental setup
- CRS models compared (from `context-aware/config.py`):
  - gemma:2b, qwen:7b, qwen:4b, llama3:instruct, llama2:latest, mistral:7b.
- User simulator model: llama3:instruct.
- LLM judge model: llama3:instruct.
- Topic extractor mode: lda by default; alternative runs exist for llm (see `context-aware/figures/llm/...`).
- Embedding backend: TF-IDF by default (`USE_TRANSFORMER_EMBEDDER=0`), sentence-transformers optional.
- Sessions per model: 1000 (counts in `context-aware/results.jsonl`).
- Turns per session: 20 user turns and 20 system turns (hard-coded in `context-aware/simulate.py`).
- Average detected topic shifts per session: ~18.8 (nearly every user turn forms a new segment).

## Primary metrics reported
- Automated adaptation metrics:
  - topic_recovery_rate, avg_recovery_delay, topic_interference,
  - cross_coherence, context_retention, context_adaptation_score (TAS/CAS).
- LLM-as-judge metrics (PEPPER-style): proactiveness, coherence, personalization.

## Key quantitative results (from `context-aware/model_metrics.csv`)
- Best topic_recovery_rate: mistral:7b (0.258), then llama2:latest (0.245).
- Fastest avg_recovery_delay (lower is better): qwen:7b (4.249), then mistral:7b (4.285).
- Lowest topic_interference (lower is better): gemma:2b (0.075), then llama3:instruct (0.096).
- Highest cross_coherence: llama3:instruct (0.114), then mistral:7b (0.100).
- Highest context_retention: llama3:instruct (0.213), then llama2:latest (0.192).
- Highest context_adaptation_score (TAS/CAS): llama3:instruct (0.189), then mistral:7b (0.159).
- LLM judge (PEPPER-style) best across all three: llama3:instruct
  - proactiveness 3.994, coherence 4.329, personalization 4.277.

## Statistical testing
- `context-aware/statistical_analysis.py` runs ANOVA and Tukey HSD.
- `context-aware/statistical_analysis_results.txt` shows significant differences for most metrics (p < 0.05).
- Use Tukey HSD tables to report which model pairs differ for each metric.

## Visualizations and what they show
- Bar charts (average per metric): compare model means for each metric.
- Distributions (KDE): show variance within each model across sessions.
- Radar charts (LLM judge): multi-metric profile of proactiveness/coherence/personalization.
- Radar charts (CAS components): tradeoffs between recovery, coherence, and retention.
- Box plot (avg_recovery_delay): spread of recovery speed across models.
- Correlation heatmap: relationships among TAS and PEPPER metrics.
- Stacked recovery bar: contrast recovery rate vs interference.
- Alignment-over-time step graph: how system alignment evolves across turns.
- System topic count distribution: how many topics systems mention per response.
- Custom plots:
  - `context-aware/visualize_context_adaptation.py` shifts scores by +0.5 for display.
  - `context-aware/visualize_pairwise_radar.py` compares two models and correlates TAS vs PEPPER.
  - `context-aware/visualize_distributions_custom.py` normalizes PEPPER metrics and plots TAS distributions.

## Notes for results and discussion
- Llama3:instruct leads in TAS/CAS and all PEPPER metrics, suggesting stronger alignment and coherence.
- Mistral:7b has the best recovery rate but weaker coherence/retention than llama3.
- Gemma:2b has the lowest interference but lower coherence and retention.
- Qwen:7b recovers fastest on average but scores low on coherence and personalization.
- Discuss tradeoffs: recovery speed vs coherence, and interference vs adaptation score.
