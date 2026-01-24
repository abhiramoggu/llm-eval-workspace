# simulate.py
# Runs a single USER ↔ CRS conversation and saves it for evaluation.

import json
import os
import random
import dataclasses
from typing import Optional

from system import CRSSystem
from user_sim import UserSimulator
from config import N_TURNS, LOG_DIR


def run_simulation(session_id: str = "001", model_name: str = "gemma:2b", seed: Optional[int] = None):
    """Run one simulated conversation and persist the log as JSON.

    Args:
        session_id: Identifier used in the log filename.
        model_name: CRS model name (for Ollama mode).
        seed: Seed for deterministic simulator policy (LLM generation may still be nondeterministic).
    """
    if seed is not None:
        random.seed(seed)

    system = CRSSystem(model_name=model_name)
    user = UserSimulator(seed=seed)

    conversation = []
    for _ in range(N_TURNS):
        user_msg = user.get_message()
        conversation.append({
            "speaker": "USER",
            "text": user_msg,
            "user_meta": user.get_last_meta()
        })

        system_reply = system.respond(user_msg)
        conversation.append({
            "speaker": "SYSTEM",
            "text": system_reply,
            "constraints": system.constraints.copy(),
            "rec_titles": system.last_rec_titles.copy(),
        })

        user.record_system_message(system_reply)

    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{model_name}_session_{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_id": session_id,
                "model_name": model_name,
                "true_genre": user.true_genre,
                "conversation": conversation,
                "shift_events": [dataclasses.asdict(e) if hasattr(e, "__dataclass_fields__") else e for e in getattr(user, "shift_events", [])],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path


if __name__ == "__main__":
    # smoke test
    log = run_simulation(session_id="debug", model_name="gemma:2b", seed=0)
    print("Saved", log)
