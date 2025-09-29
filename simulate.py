# simulate.py
import json
import os
from system import CRSSystem
from user_sim import UserSimulator
from evaluate import evaluate, llm_judge
from config import LLM_SYSTEMS

LOG_DIR = "logs"

def run_simulation(session_id="001", model_name="gemma:2b"):
    system = CRSSystem(model_name=model_name)
    user = UserSimulator()

    conversation = []
    for turn in range(3):
        user_msg = user.get_message(turn)
        conversation.append({"speaker": "USER", "text": user_msg})
        print("USER:", user_msg)

        system_reply = system.respond(user_msg)
        conversation.append({"speaker": "SYSTEM", "text": system_reply, "constraints": system.constraints.copy()})
        print("SYSTEM:", system_reply)

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
    print("LLM-JUDGE EVALUATION:", llm_judge(convo))
