# user_sim.py
# LLM that simulates a human user who asks questions, reacts, and changes topics

import random, subprocess
from config import MODE, USER_MODEL, ERROR_RATE

class UserSimulator:
    def __init__(self):
        self.true_genre = random.choice(["romance", "horror", "sci-fi"])
        self.current_topic = self.true_genre
        self.error_turns = sorted(random.sample(range(4, 20), k=3))  # ~3 topic shifts
        self.turn_count = 0
        self.conversation_history = []
        self.error_injected = False

    def call_ollama(self, prompt: str) -> str:
        """Call Ollama for user simulation text generation."""
        if MODE == "mock":
            return "I'm thinking about watching something fun today."
        process = subprocess.Popen(
            ["ollama", "run", USER_MODEL],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        )
        output, _ = process.communicate(input=prompt)
        return output.strip()

    def inject_topic_change(self):
        """Pick a new genre to simulate topic shift."""
        choices = ["romance", "horror", "sci-fi"]
        choices.remove(self.current_topic)
        new_genre = random.choice(choices)
        return new_genre

    def get_message(self):
        """Simulate each user turn."""
        turn = self.turn_count
        self.turn_count += 1

        # --- turn 0: express preference (maybe wrong)
        if turn == 0:
            if random.random() < ERROR_RATE:
                wrong = self.inject_topic_change()
                self.error_injected = True
                msg = f"I want to watch a {wrong} movie."
                self.current_topic = wrong
            else:
                msg = f"I want to watch a {self.true_genre} movie."

        # --- later turns: conversation flow and topic shifts
        else:
            # gradually more likely to change topic
            if turn in self.error_turns:
                new_genre = self.inject_topic_change()
                self.current_topic = new_genre
                msg = f"Hmm, maybe I’m actually in the mood for a {new_genre} movie instead."
            else:
                convo = "\n".join([f"{t['speaker']}: {t['text']}" for t in self.conversation_history])
                prompt = f"""
You are the USER in a movie recommendation chat.
Here is the conversation so far:

{convo}

Now respond as the USER would in 1–2 sentences:
- Ask about the recommended movies, actors, or themes.
- Express emotions (interest, doubt, confusion, excitement).
- Occasionally react to what the assistant said earlier.
- Do NOT recommend movies yourself.
"""
                msg = self.call_ollama(prompt)

        self.conversation_history.append({"speaker": "USER", "text": msg})
        return msg

    def record_system_message(self, message: str):
        """Store the CRS response so future turns can reference it."""
        self.conversation_history.append({"speaker": "SYSTEM", "text": message})
