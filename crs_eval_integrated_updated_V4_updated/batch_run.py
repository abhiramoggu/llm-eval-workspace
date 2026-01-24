# batch_run.py
"""
Phase 1: Run all simulations and save conversation logs.
Phase 2: Evaluate all saved logs quantitatively and with an LLM-judge.
Phase 3: Aggregate results into a summary CSV.
"""

import json
import pandas as pd
import os
import shutil
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from config import LLM_SYSTEMS, RESULTS_FILE, SUMMARY_CSV, LOG_DIR, FIG_DIR, N_SESSIONS
from simulate import run_simulation
from evaluate import evaluate, llm_judge


def _clear_previous_run_data():
    """Wipes old logs/results/figures to ensure a clean run."""
    print("--- Clearing previous run data ---")

    if os.path.exists(LOG_DIR):
        for filename in os.listdir(LOG_DIR):
            # Clear session logs and per-model judge scores
            if filename.endswith(".json") or filename.endswith("_judge_scores.csv"):
                os.remove(os.path.join(LOG_DIR, filename))

    for p in [RESULTS_FILE, SUMMARY_CSV]:
        if os.path.exists(p):
            os.remove(p)

    if os.path.exists(FIG_DIR):
        shutil.rmtree(FIG_DIR, ignore_errors=True)
        os.makedirs(FIG_DIR, exist_ok=True)

def _write_summaries(rows):
    # --- Phase 3: Aggregate results to CSV ---
    print("\n--- Phase 3: Aggregating Results ---")
    df = pd.DataFrame(rows)
    if df.empty:
        print("No results to summarize.")
        return
    judge_df = pd.json_normalize(df["judge"]) if "judge" in df.columns else pd.DataFrame()
    if not judge_df.empty:
        df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)
    else:
        df = df.drop(columns=["judge"], errors="ignore")

    metric_cols = [
        "topic_recovery_rate", "avg_recovery_delay", "topic_interference", "cross_coherence",
        "context_retention", "context_adaptation_score", "proactiveness", "coherence",
        "personalization"
    ]
    summary_cols = [col for col in metric_cols if col in df.columns]
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


def run_batch(n_sessions=N_SESSIONS):
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
        # Paired sessions: for each session index i, run *all* models using the same simulator seed.
        # This enables paired statistical tests (Wilcoxon/paired t-test) across models.
        for i in range(n_sessions):
            for model in LLM_SYSTEMS:
                session_id = f"{i:05d}"
                log_path = run_simulation(session_id=session_id, model_name=model, seed=i)
                log_files.append(log_path)
                progress.advance(sim_tasks[model])

        # --- Phase 2: Evaluate all saved logs --- all saved logs ---
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

                eval_out = evaluate(data, genre)
                judge = llm_judge(convo)
                record = {
                    "model": model_name,
                    "session_id": os.path.basename(log_path).split("_session_")[-1].replace(".json",""),
                    **eval_out, # Keep the 'detail' dictionary
                    "judge": judge
                }
                f.write(json.dumps(record) + "\n")
                progress.advance(eval_tasks[model_name])

    rows = [json.loads(line) for line in open(RESULTS_FILE)]
    _write_summaries(rows)


if __name__ == "__main__":
    run_batch()
