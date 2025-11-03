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


def run_batch(n_sessions=200):
    # Setup a single Rich Progress instance for a multi-line display
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
        "context_retention", "context_adaptation_score", "clarity", "politeness", 
        "recovery", "context_memory", "engagement"
    ]
    # Filter out columns that might not exist to prevent KeyErrors
    summary_cols = [col for col in metric_cols if col in df.columns]
    summary = df.groupby("model")[summary_cols].mean().reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary → {SUMMARY_CSV}")


if __name__ == "__main__":
    run_batch(n_sessions=200) # Reduced for quick testing
