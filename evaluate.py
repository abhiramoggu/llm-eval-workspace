# evaluate.py
import subprocess
import json

def evaluate(conversation, true_genre):
    """
    Rule-based evaluation for error recovery in CRS.
    - Did the system ever mention the true genre in its response?
    - How many turns until it recovered?
    """
    recovery_success = False
    recovery_turns = None

    for i, turn in enumerate(conversation):
        if turn["speaker"] == "SYSTEM":
            if true_genre in turn["text"].lower():
                recovery_success = True
                recovery_turns = i // 2  # system speaks every 2nd turn
                break

    return {
        "true_genre": true_genre,
        "recovery_success": recovery_success,
        "recovery_turns": recovery_turns
    }


def llm_judge(conversation, model_name="llama2"):
    """
    Ask an Ollama model to judge the conversation quality.
    Metrics: clarity, politeness, recovery quality.
    Returns a dictionary of scores (0-5).
    """
    # Format conversation into a single text block
    convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in conversation])

    # Build the judging prompt
    prompt = f"""
You are an evaluator. Here is a dialogue between a USER and a SYSTEM:

{convo_text}

Please rate the SYSTEM on these metrics (0-5 scale, 5 is best):
- Clarity (was the response understandable?)
- Politeness (was it respectful and natural?)
- Recovery (did it handle misunderstandings or corrections well?)

Return JSON like this:
{{"clarity": X, "politeness": Y, "recovery": Z}}
    """

    # Call Ollama model
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode("utf-8"),
        capture_output=True,
    )

    output = result.stdout.decode("utf-8").strip()

    # Try to parse JSON from model output
    try:
        scores = json.loads(output)
    except:
        scores = {"clarity": None, "politeness": None, "recovery": None}

    return scores
