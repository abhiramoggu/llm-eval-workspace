# user_sim.py
import random
import subprocess
from config import MODE, USER_MODEL, ERROR_RATE

class UserSimulator:
    def __init__(self):
        self.true_genre = random.choice(["romance", "horror", "sci-fi"])
        self.error_injected = False
        self.conversation_history = []

    def call_ollama(self, prompt: str) -> str:
        if MODE == "mock":
            # Fallback mock
            return "USER: I'd like a movie."
        else:
            process = subprocess.Popen(
                ["ollama", "run", USER_MODEL],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            output, error = process.communicate(input=prompt)
            if error:
                print("Error:", error)
            return output.strip()

    def get_message(self, turn: int) -> str:
        """
        Use LLM to simulate user turns.
        - Turn 0: express preference (sometimes wrong).
        - Turn 1: correct if wrong.
        - Later: respond naturally.
        """
        if turn == 0:
            if random.random() < ERROR_RATE:
                wrong_genre = random.choice(["romance", "horror", "sci-fi"])
                while wrong_genre == self.true_genre:
                    wrong_genre = random.choice(["romance", "horror", "sci-fi"])
                self.error_injected = True
                user_msg = f"I want to watch a {wrong_genre} movie."
            else:
                user_msg = f"I want to watch a {self.true_genre} movie."

        elif turn == 1 and self.error_injected:
            user_msg = f"Oops, I meant {self.true_genre}, not the other genre."
        else:
            # Let the LLM produce a natural follow-up
            convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in self.conversation_history])
            user_prompt = fuser_prompt = f"""
You are an ordinary USER having a natural conversation with a movie recommender assistant.

- Your goal is to eventually get a good movie recommendation, but you may express preferences gradually or change them mid-way.
- You may start with one genre, but later realize you meant a different one.
- You may switch topics briefly, ask about actors, change your mind, or even go off-topic for a message or two.
- Try to sound like a normal human — casual, curious, occasionally confused, polite but not robotic.
- You should simulate at least 3 shifts in topic, interest, or correction during the full conversation (which will be around 20 messages long).
- Do NOT just repeat the same sentence structure or state your genre every time.

The system will evaluate how the assistant recovers from your mistakes or shifting preferences.

Here is the conversation so far:

{convo_text}

Respond now with what the USER would naturally say next, in 1–2 sentences.
"""

            user_msg = self.call_ollama(user_prompt)

        # Update history
        self.conversation_history.append({"speaker": "USER", "text": user_msg})
        return user_msg
