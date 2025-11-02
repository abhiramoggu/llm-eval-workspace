# batch_run.py
import os
import json
import csv
from simulate import run_simulation
from evaluate import evaluate, llm_judge
from config import LLM_SYSTEMS

RESULTS_DIR = "results"

def run_batch(models, n_sessions=500):
    """
    Run multiple simulations for multiple CRS models.
    Collect rule-based (text/state/recs) and LLM-judge metrics.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for model in models:
        print(f"\n=== Running batch for {model} ===")
        model_results = []

        for i in range(n_sessions):
            session_id = f"{model}_session_{i+1}"
            convo, true_genre = run_simulation(session_id=session_id, model_name=model)

            # Rule-based upgraded evaluation
            rule_scores = evaluate(convo, true_genre)

            # LLM judge evaluation
            llm_scores = llm_judge(convo)

            result_entry = {
                "session_id": session_id,
                "model": model,
                "true_genre": true_genre,
                "text_recovery_success": rule_scores["text_recovery_success"],
                "state_recovery_success": rule_scores["state_recovery_success"],
                "rec_consistency": rule_scores["rec_consistency"],
                "recovery_turns": rule_scores["recovery_turns"],
                "clarity": llm_scores.get("clarity"),
                "politeness": llm_scores.get("politeness"),
                "recovery": llm_scores.get("recovery")
            }
            model_results.append(result_entry)

            print(f"Session {i+1}/{n_sessions} → {rule_scores}, {llm_scores}")

        # Save per-model results to JSON
        result_path = os.path.join(RESULTS_DIR, f"{model}_results.json")
        with open(result_path, "w") as f:
            json.dump(model_results, f, indent=2)

        all_results[model] = model_results

    # Save all sessions into a single CSV
    csv_path = os.path.join(RESULTS_DIR, "batch_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "session_id", "model", "true_genre",
            "text_recovery_success", "state_recovery_success",
            "rec_consistency", "recovery_turns",
            "clarity", "politeness", "recovery"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model, sessions in all_results.items():
            for entry in sessions:
                writer.writerow(entry)

    print(f"\nResults saved to {RESULTS_DIR}/ (JSON + CSV).")
    return all_results


if __name__ == "__main__":
    results = run_batch(models=LLM_SYSTEMS, n_sessions=500)

    print("\n=== SUMMARY ===")
    for model, sessions in results.items():
        text_rate = sum(s["text_recovery_success"] for s in sessions) / len(sessions)
        state_rate = sum(s["state_recovery_success"] for s in sessions) / len(sessions)
        rec_rate = sum(s["rec_consistency"] for s in sessions) / len(sessions)
        avg_turns = sum((s["recovery_turns"] or 0) for s in sessions) / len(sessions)

        print(f"\nModel: {model}")
        print(f"  Text Recovery Rate: {text_rate:.2f}")
        print(f"  State Recovery Rate: {state_rate:.2f}")
        print(f"  Recommendation Consistency: {rec_rate:.2f}")
        print(f"  Avg Recovery Turns: {avg_turns:.2f}")
