# run_evaluations.py
# Evaluates existing logs and produces results/plots inputs without rerunning simulations.

import json
import os
import pandas as pd

from config import RESULTS_FILE, SUMMARY_CSV, LOG_DIR
from evaluate import evaluate, llm_judge


def _collect_log_files(log_dir: str):
    return sorted(
        os.path.join(log_dir, name)
        for name in os.listdir(log_dir)
        if name.endswith(".json") and "_session_" in name
    )


def _write_summaries(rows):
    print("\n--- Aggregating Results ---")
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

    judge_cols = ["proactiveness", "coherence", "personalization"]
    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        per_model_judge_path = os.path.join(LOG_DIR, f"{model_name}_judge_scores.csv")
        model_df[["model"] + judge_cols].to_csv(per_model_judge_path, index=False)
        print(f"Saved per-model judge scores → {per_model_judge_path}")
    print(f"\nSaved summary → {SUMMARY_CSV}")


def run_evaluations():
    log_files = _collect_log_files(LOG_DIR)
    if not log_files:
        print("No log files found to evaluate.")
        return

    rows = []
    with open(RESULTS_FILE, "w") as f:
        for log_path in log_files:
            with open(log_path, "r") as log_file:
                data = json.load(log_file)
            convo = data["conversation"]
            genre = data.get("true_genre")
            model_name = data.get("model_name", os.path.basename(log_path).split("_session_")[0])

            eval_out = evaluate(data, genre)
            judge = llm_judge(convo)
            record = {
                "model": model_name,
                "session_id": os.path.basename(log_path).split("_session_")[-1].replace(".json", ""),
                **eval_out,
                "judge": judge,
            }
            rows.append(record)
            f.write(json.dumps(record) + "\n")

    _write_summaries(rows)


if __name__ == "__main__":
    run_evaluations()
