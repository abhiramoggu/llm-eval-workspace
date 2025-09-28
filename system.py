# system.py
from dataset import recommend

class CRSSystem:
    def __init__(self, model_name="mock-llm"):
        self.model_name = model_name
        self.constraints = {}

    def call_ollama(self, prompt: str) -> str:
        """
        MOCK FUNCTION for now:
        Pretend the LLM just returns a genre keyword.
        Replace this later with a real ollama subprocess call.
        """
        if "Extract" in prompt:
            # crude extraction rule
            if "romance" in prompt.lower():
                return "romance"
            elif "horror" in prompt.lower():
                return "horror"
            elif "sci-fi" in prompt.lower():
                return "sci-fi"
            else:
                return "romance"  # default fallback
        else:
            return "Here are some movies you might like!"

    def respond(self, user_message: str) -> str:
        # Step 1: Extract genre
        extraction_prompt = f"User said: '{user_message}'. Extract the genre."
        extracted = self.call_ollama(extraction_prompt)
        self.constraints["genre"] = extracted

        # Step 2: Get recs
        recs = recommend(self.constraints)
        rec_titles = [m["title"] for m in recs]

        # Step 3: Generate reply
        reply_prompt = (
            f"The user wants {self.constraints['genre']} movies. "
            f"Recommend from {rec_titles}."
        )
        reply = self.call_ollama(reply_prompt)
        return reply
