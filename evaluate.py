# evaluate.py
import subprocess
import json
from config import MODE, JUDGE_MODEL

def evaluate(conversation, true_genre):
    recovery_success = False
    recovery_turns = None

    for i, turn in enumerate(conversation):
        if turn["speaker"] == "SYSTEM":
            if true_genre in turn["text"].lower():
                recovery_success = True
                recovery_turns = i // 2
                break

    return {
        "true_genre": true_genre,
        "recovery_success": recovery_success,
        "recovery_turns": recovery_turns
    }


def llm_judge(conversation):
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
