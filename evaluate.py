# evaluate.py
import subprocess
import json
from config import MODE, JUDGE_MODEL

def evaluate(conversation, true_genre):
    """
    Upgraded evaluation:
    - text_recovery_success: Did system mention the true genre in its reply?
    - state_recovery_success: Did system.constraints get updated to true_genre?
    - rec_consistency: Did the recommended items match true_genre?
    - recovery_turns: Number of turns until recovery.
    """

    text_recovery_success = False
    state_recovery_success = False
    rec_consistency = False
    recovery_turns = None

    for i, turn in enumerate(conversation):
        if turn["speaker"] == "SYSTEM":
            # Text check 
            # this is responsible for checking the how the system recovers in dialogue if it makes a mistake based on the true genre
            if true_genre in turn["text"].lower() and not text_recovery_success:
                text_recovery_success = True
                recovery_turns = i // 2  # system speaks every 2nd turn

            # State check: constraints snapshot included in simulate.py
            constraints = turn.get("constraints", {})
            if constraints.get("genre") == true_genre:
                state_recovery_success = True

            # Recommendation check (if items present in reply)
            # Simple heuristic: look for a movie from the true genre
            # (Your dataset knows genres, but here we just do a string match for simplicity)
            if true_genre in turn["text"].lower():
                rec_consistency = True

    return {
        "true_genre": true_genre,
        "text_recovery_success": text_recovery_success,
        "state_recovery_success": state_recovery_success,
        "rec_consistency": rec_consistency,
        "recovery_turns": recovery_turns
    }


def llm_judge(conversation):
    """
    LLM-as-judge evaluation for subjective metrics.
    """
    convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in conversation])

    prompt = f"""
You are an evaluator. Here is a dialogue between a USER and a SYSTEM:

{convo_text}

Please rate the SYSTEM on these metrics (0-5 scale, 5 is best):
- Clarity
- Politeness
- Recovery ability

Return JSON like this:
{{"clarity": X, "politeness": Y, "recovery": Z}}
    """

    if MODE == "mock":
        return {"clarity": 4, "politeness": 5, "recovery": 3}
    else:
        process = subprocess.Popen(
            ["ollama", "run", JUDGE_MODEL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=prompt)
        if error:
            print("Error:", error)

        try:
            return json.loads(output.strip())
        except:
            return {"clarity": None, "politeness": None, "recovery": None}
