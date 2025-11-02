# system.py
# LLMs that act as CRS — receive user input, extract intent, recommend movies

import subprocess
from dataset import recommend
from config import MODE

class CRSSystem:
    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        self.constraints = {}

    @staticmethod
    def _normalize_genre(raw_genre: str, fallback: str | None = None) -> str:
        """Map LLM extraction output onto known genres."""
        if not raw_genre:
            return fallback or "romance"

        text = raw_genre.lower()
        if "sci" in text or "science fiction" in text:
            return "sci-fi"
        for genre in ("romance", "horror", "sci-fi"):
            if genre in text:
                return genre
        if "science-fiction" in text:
            return "sci-fi"
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
        """Extract genre intent, fetch recommendations, generate reply."""
        extraction_prompt = f"Extract the genre (romance, horror, sci-fi) from: '{user_message}'."
        extracted = self.call_ollama(extraction_prompt)

        previous_genre = self.constraints.get("genre")
        normalized_genre = self._normalize_genre(extracted, fallback=previous_genre)
        self.constraints["genre"] = normalized_genre

        recs = recommend(self.constraints)
        rec_titles = [m["title"] for m in recs]
        reply_prompt = (
            f"The user wants {self.constraints['genre']} movies. "
            f"Recommend politely and conversationally from these: {rec_titles}. "
            "Maintain coherence with prior context."
        )
        reply = self.call_ollama(reply_prompt)
        return reply
