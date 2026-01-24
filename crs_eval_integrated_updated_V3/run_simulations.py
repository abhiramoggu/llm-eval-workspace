# run_simulations.py
# Runs simulations only (always clears old logs/results first).

import os
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from config import LLM_SYSTEMS, LOG_DIR, RESULTS_FILE
from simulate import run_simulation


def _clear_previous_run_data():
    """Wipes old logs and results to ensure a clean run."""
    print("--- Clearing previous run data ---")
    if os.path.exists(LOG_DIR):
        for filename in os.listdir(LOG_DIR):
            if filename.endswith(".json") or filename.endswith("_judge_scores.csv"):
                os.remove(os.path.join(LOG_DIR, filename))
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)


def run_simulations(n_sessions=1000):
    _clear_previous_run_data()
    progress = Progress(
        TextColumn("[bold cyan]Simulating [bold white]→[/] {task.description}", justify="right"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        tasks = {
            model: progress.add_task(model, total=n_sessions)
            for model in LLM_SYSTEMS
        }
        for i in range(n_sessions):
            for model in LLM_SYSTEMS:
                session_id = f"{i:05d}"
                run_simulation(session_id=session_id, model_name=model, seed=i)
                progress.advance(tasks[model])


if __name__ == "__main__":
    run_simulations(n_sessions=1000)
