# simulate.py
# Runs a single user ↔ CRS conversation and saves it for evaluation

import json, os
from system import CRSSystem
from user_sim import UserSimulator
from evaluate import evaluate, llm_judge
from config import LLM_SYSTEMS

LOG_DIR = "logs"

def run_simulation(session_id="001", model_name="gemma:2b"):
    system = CRSSystem(model_name=model_name)
    user = UserSimulator()
    conversation = []

    for turn in range(20):
        user_msg = user.get_message()
        conversation.append({"speaker": "USER", "text": user_msg})
        print("USER:", user_msg)

        system_reply = system.respond(user_msg)
        conversation.append({
            "speaker": "SYSTEM",
            "text": system_reply,
            "constraints": system.constraints.copy()
        })
        print("SYSTEM:", system_reply)

        user.record_system_message(system_reply)

    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{model_name}_session_{session_id}.json")
    with open(path, "w") as f:
        json.dump({"true_genre": user.true_genre, "conversation": conversation}, f, indent=2)
    print(f"\nSaved: {path}")

    return conversation, user.true_genre

if __name__ == "__main__":
    convo, genre = run_simulation()
    results = evaluate(convo, genre)
    print("Quantitative:", results)
    print("LLM-Judge:", llm_judge(convo))
