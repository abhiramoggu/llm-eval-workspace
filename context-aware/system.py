# system.py
# LLMs that act as CRS — receive user input, extract intent, recommend movies

import subprocess
from dataset import recommend
from config import MODE

class CRSSystem:
    KNOWN_GENRES = {"romance", "horror", "sci-fi", "action", "comedy", "drama", "thriller", "fantasy", "animation", "crime", "mystery", "adventure", "biopic", "musical", "historical"}

    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        self.constraints = {}

    @staticmethod
    def _normalize_genre(raw_genre: str, fallback: str | None = None) -> str:
        """Map extraction output onto known genres. More robustly checks for substrings."""
        if not raw_genre:
            return fallback or "romance"

        text = raw_genre.lower()
        # Check for specific genres first
        for genre in CRSSystem.KNOWN_GENRES:
            if genre in text:
                return genre
        return fallback or "romance"

    def call_ollama(self, prompt: str):
        """Call Ollama or mock for quick testing."""
        if MODE == "mock":
            return "Sure! You might like 'Titanic' or 'Inception'."
        process = subprocess.Popen(
            ["ollama", "run", self.model_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        )
        output, _ = process.communicate(input=prompt)
        return output.strip()

    def respond(self, user_message: str) -> str:
        """
        A more robust, single-step response generation.
        1. Extract genre from user message heuristically.
        2. Fetch recommendations based on the extracted genre.
        3. Generate a conversational reply using the recommendations as context.
        """
        # Step 1: Heuristic extraction is faster and good enough for this stage.
        # We rely on the final generation LLM to be conversational.
        extracted_genre = self._normalize_genre(user_message, fallback=self.constraints.get("genre"))
        self.constraints["genre"] = extracted_genre

        # Step 2: Fetch recommendations.
        recs = recommend(self.constraints)
        rec_titles = [m["title"] for m in recs]

        # Step 3: Generate a high-quality, conversational reply in one shot.
        reply_prompt = (
            f"You are a helpful and friendly movie recommender. "
            f"The user just said: '{user_message}'. "
            f"Based on their message, they seem interested in the '{self.constraints['genre']}' genre. "
            f"Politely and conversationally recommend one or two movies from this list: {rec_titles}. "
            f"Keep your response to 1-2 sentences."
        )
        reply = self.call_ollama(reply_prompt)
        return reply
