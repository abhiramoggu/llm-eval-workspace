# batch_run.py
# Run multiple sessions automatically for all CRS models

from simulate import run_simulation
from evaluate import evaluate, llm_judge
from config import LLM_SYSTEMS
import json

RESULTS_FILE = "results.jsonl"

def run_batch(n_sessions=5):
    with open(RESULTS_FILE, "w") as f:
        for model in LLM_SYSTEMS:
            print(f"\n=== Running {n_sessions} sessions for {model} ===")
            for i in range(n_sessions):
                convo, genre = run_simulation(f"{model}_{i}", model)
                metrics = evaluate(convo, genre)
                judge = llm_judge(convo)
                record = {"model": model, **metrics, "judge": judge}
                f.write(json.dumps(record) + "\n")
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_batch()
