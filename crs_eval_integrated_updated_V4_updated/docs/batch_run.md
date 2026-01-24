# batch_run.py

Runs experiments across multiple CRS models.

## Key idea: paired sessions
For each session index `i`, it runs *all models* using the same simulator seed `i`.  
This enables paired statistical tests across models (Wilcoxon / paired t-test).

## Outputs
- conversation logs in `logs/`
- aggregated metrics in `results.jsonl`
- summary CSV in `model_metrics.csv`
- per-model judge score CSVs in `logs/<model>_judge_scores.csv`

## Important behavior
- The script clears previous `logs/` JSON files and `results.jsonl` at startup.

## Typical usage
```bash
python -c "from batch_run import run_batch; run_batch(n_sessions=100)"
python analyze_results.py
```

You can also set `N_SESSIONS` in `config.py` or export `N_SESSIONS` in the shell.
