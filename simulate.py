# simulate.py
import json
import os
from system import CRSSystem
from user_sim import UserSimulator
from evaluate import evaluate, llm_judge

LOG_DIR = "logs"

def run_simulation(session_id="001"):
    system = CRSSystem(model_name="mock-llm")
    user = UserSimulator()

    conversation = []
    for turn in range(3):
        user_msg = user.get_message(turn)
        conversation.append({"speaker": "USER", "text": user_msg})
        print("USER:", user_msg)

        system_reply = system.respond(user_msg)
        conversation.append({
            "speaker": "SYSTEM",
            "text": system_reply,
            "constraints": system.constraints.copy()
        })
        print("SYSTEM:", system_reply)

    # Save log
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"session_{session_id}.json")
    with open(log_path, "w") as f:
        json.dump({"true_genre": user.true_genre, "conversation": conversation}, f, indent=2)

    print(f"\nConversation saved to {log_path}")
    return conversation, user.true_genre

if __name__ == "__main__":
    convo, true_genre = run_simulation()
    scores = evaluate(convo, true_genre)
    print("\nRULE-BASED EVALUATION:", scores)

    # LLM judge evaluation (comment out if Ollama not running)
    llm_scores = llm_judge(convo, model_name="llama2")
    print("LLM-JUDGE EVALUATION:", llm_scores)
