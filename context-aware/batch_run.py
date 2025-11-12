# batch_run.py
"""
Phase 1: Run all simulations and save conversation logs.
Phase 2: Evaluate all saved logs quantitatively and with an LLM-judge.
Phase 3: Aggregate results into a summary CSV.
"""

import json
import pandas as pd
import os
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from config import LLM_SYSTEMS, RESULTS_FILE, SUMMARY_CSV, LOG_DIR
from simulate import run_simulation
from evaluate import evaluate, llm_judge


def _clear_previous_run_data():
    """Wipes old logs and results to ensure a clean run."""
    print("--- Clearing previous run data ---")
    if os.path.exists(LOG_DIR):
        for filename in os.listdir(LOG_DIR):
            # Clear session logs and per-model judge scores
            if filename.endswith(".json") or filename.endswith("_judge_scores.csv"):
                os.remove(os.path.join(LOG_DIR, filename))
    
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

def run_batch(n_sessions=50):
    # Setup a single Rich Progress instance for a multi-line display
    _clear_previous_run_data()
    progress = Progress(
        TextColumn("[bold cyan]{task.fields[phase]} [bold white]→[/] {task.description}", justify="right"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with Live(progress, auto_refresh=True, vertical_overflow="visible") as live:
        # --- Phase 1: Run all simulations ---
        sim_tasks = {
            model: progress.add_task(model, total=n_sessions, phase="Simulating")
            for model in LLM_SYSTEMS
        }

        log_files = []
        for model in LLM_SYSTEMS:
            for i in range(n_sessions):
                session_id = f"{model}_{i}"
                log_path = run_simulation(session_id, model_name=model)
                log_files.append(log_path)
                progress.advance(sim_tasks[model])

        # --- Phase 2: Evaluate all saved logs ---
        eval_tasks = {
            model: progress.add_task(model, total=n_sessions, phase="Evaluating", visible=False)
            for model in LLM_SYSTEMS
        }

        with open(RESULTS_FILE, "w") as f:
            for log_path in log_files:
                with open(log_path, 'r') as log_file:
                    data = json.load(log_file)
                convo = data["conversation"]
                genre = data["true_genre"]
                model_name = data.get("model_name", os.path.basename(log_path).split('_session_')[0])

                # Make the correct progress bar visible
                progress.update(sim_tasks[model_name], visible=False)
                progress.update(eval_tasks[model_name], visible=True)

                eval_out = evaluate(convo, genre)
                judge = llm_judge(convo)
                record = {
                    "model": model_name,
                    **eval_out, # Keep the 'detail' dictionary
                    "judge": judge
                }
                f.write(json.dumps(record) + "\n")
                progress.advance(eval_tasks[model_name])

    # --- Phase 3: Aggregate results to CSV ---
    print("\n--- Phase 3: Aggregating Results ---")
    rows = [json.loads(line) for line in open(RESULTS_FILE)]
    df = pd.DataFrame(rows)
    judge_df = pd.json_normalize(df["judge"])
    df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)

    metric_cols = [
        "topic_recovery_rate", "avg_recovery_delay", "topic_interference", "cross_coherence", 
        "context_retention", "context_adaptation_score", "proactiveness", "coherence", 
        "personalization"
    ]
    # Filter out columns that might not exist to prevent KeyErrors
    summary_cols = [col for col in metric_cols if col in df.columns]
    # Ensure columns are numeric, coercing errors to NaN which mean() will ignore
    for col in summary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    summary = df.groupby("model")[summary_cols].mean().reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)

    # --- Phase 4: Save per-model judge scores ---
    judge_cols = ["proactiveness", "coherence", "personalization"]
    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        per_model_judge_path = os.path.join(LOG_DIR, f"{model_name}_judge_scores.csv")
        model_df[['model'] + judge_cols].to_csv(per_model_judge_path, index=False)
        print(f"Saved per-model judge scores → {per_model_judge_path}")
    print(f"\nSaved summary → {SUMMARY_CSV}")


if __name__ == "__main__":
    run_batch(n_sessions=50) # Reduced for quick testing
