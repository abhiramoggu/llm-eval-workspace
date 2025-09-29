# system.py
import subprocess
from dataset import recommend
from config import MODE

class CRSSystem:
    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        self.constraints = {}

    def call_ollama(self, prompt: str) -> str:
        """
        Calls Ollama if MODE=ollama, else returns mock strings.
        """
        if MODE == "mock":
            # Simple mock for testing
            if "romance" in prompt.lower():
                return "romance"
            elif "horror" in prompt.lower():
                return "horror"
            elif "sci-fi" in prompt.lower():
                return "sci-fi"
            else:
                return "Here are some movies you might like!"
        else:
            # Real Ollama call
            process = subprocess.Popen(
                ["ollama", "run", self.model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            output, error = process.communicate(input=prompt)
            if error:
                print("Error:", error)
            return output.strip()

    def respond(self, user_message: str) -> str:
        """
        1. Extract constraints with LLM.
        2. Query recommender.
        3. Generate reply.
        """
        extraction_prompt = f"User said: '{user_message}'. Extract the genre (romance, horror, sci-fi)."
        extracted = self.call_ollama(extraction_prompt)
        self.constraints["genre"] = extracted.lower()

        recs = recommend(self.constraints)
        rec_titles = [m["title"] for m in recs]

        reply_prompt = (
            f"The user wants {self.constraints['genre']} movies. "
            f"Recommend politely from these options: {rec_titles}."
        )
        reply = self.call_ollama(reply_prompt)
        return reply
