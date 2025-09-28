# batch_run.py
import os
import json
from simulate import run_simulation
from evaluate import evaluate, llm_judge

RESULTS_DIR = "results"

def run_batch(models, n_sessions=5):
    """
    Run multiple simulations for multiple CRS models.
    Args:
        models: list of system model names (e.g., ["mock-llm", "gemma:2b"])
        n_sessions: how many simulated conversations per model
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for model in models:
        print(f"\n=== Running batch for {model} ===")
        model_results = []

        for i in range(n_sessions):
            session_id = f"{model}_session_{i+1}"
            convo, true_genre = run_simulation(session_id=session_id)

            # Rule-based evaluation
            rule_scores = evaluate(convo, true_genre)

            # LLM judge (comment out if Ollama not available)
            llm_scores = llm_judge(convo, model_name="llama2")

            # Collect all
            result_entry = {
                "session_id": session_id,
                "true_genre": true_genre,
                "rule_scores": rule_scores,
                "llm_scores": llm_scores
            }
            model_results.append(result_entry)

            print(f"Session {i+1}/{n_sessions} complete → {rule_scores}, {llm_scores}")

        # Save model-level results
        result_path = os.path.join(RESULTS_DIR, f"{model}_results.json")
        with open(result_path, "w") as f:
            json.dump(model_results, f, indent=2)

        all_results[model] = model_results

    return all_results


if __name__ == "__main__":
    models_to_test = ["mock-llm"]  # extend later with ["gemma:2b", "qwen:4b", "deepseek:7b"]
    results = run_batch(models=models_to_test, n_sessions=3)

    print("\n=== SUMMARY ===")
    for model, sessions in results.items():
        recovery_rate = sum(s["rule_scores"]["recovery_success"] for s in sessions) / len(sessions)
        avg_recovery_turns = sum(
            s["rule_scores"]["recovery_turns"] or 0 for s in sessions
        ) / len(sessions)

        print(f"\nModel: {model}")
        print(f"  Recovery Rate: {recovery_rate:.2f}")
        print(f"  Avg Recovery Turns: {avg_recovery_turns:.2f}")
