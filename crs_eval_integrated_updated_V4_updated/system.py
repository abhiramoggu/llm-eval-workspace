# system.py
# LLMs that act as CRS — receive user input, extract intent, recommend movies

import re
import subprocess
from typing import Dict, List

from dataset import (
    recommend,
    _normalize_genre_name,
    find_attribute_in_text,
    find_plot_keywords_in_text,
)
from config import MODE, KNOWN_GENRES

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

class CRSSystem:
    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        self.constraints = {}
        self.last_rec_titles: List[str] = []

    @staticmethod
    def _normalize_genre(raw_genre: str, fallback: str | None = None) -> str:
        """Map extraction output onto known genres. More robustly checks for substrings."""
        if not raw_genre:
            return fallback or "romance"

        text = raw_genre.lower()
        # Check for specific genres first by checking for substrings
        for genre in KNOWN_GENRES:
            if genre in text:
                return genre
        return fallback or "drama" # A safe, broad default

    def call_ollama(self, prompt: str):
        """Call Ollama or mock for quick testing."""
        if MODE == "mock":
            return "Sure! You might like 'Titanic' or 'Inception'."
        process = subprocess.Popen(
            ["ollama", "run", self.model_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        )
        output, error = process.communicate(input=prompt)
        err_text = (error or "").strip()
        if err_text and any(kw in err_text.lower() for kw in ["error", "failed"]):
            print(f"Ollama Error ({self.model_name}): {err_text}")
        return (output or "").strip()

    def _extract_constraints(self, user_message: str) -> Dict[str, object]:
        """Pull out attribute cues (genre, year, actor, etc.) from free-form text."""
        lowered = user_message.lower()
        extracted: Dict[str, object] = {}
        genre = self._normalize_genre(user_message, fallback=self.constraints.get("genre"))
        if genre:
            extracted["genre"] = genre

        if match := YEAR_PATTERN.search(user_message):
            extracted["year"] = match.group()

        for attr in ("actor", "director", "language", "writer", "name"):
            if value := find_attribute_in_text(attr, lowered):
                extracted[attr] = value

        plot_keywords = find_plot_keywords_in_text(user_message)
        if plot_keywords:
            extracted["plot"] = plot_keywords[:2]

        # If the user references a specific movie title, treat it as a name constraint.
        if "name" in extracted:
            extracted.setdefault("title", extracted["name"])
        return extracted

    def _update_constraints(self, user_message: str):
        """Merge new constraints and occasionally reset when user loosens requirements."""
        lowered = user_message.lower()
        if any(kw in lowered for kw in ["anything works", "whatever", "surprise me", "anything really"]):
            self.constraints = {"genre": self.constraints.get("genre")}
            return

        extracted = self._extract_constraints(user_message)
        if not extracted:
            return

        if "genre" in extracted:
            self.constraints["genre"] = extracted["genre"]

        for key in ("year", "actor", "director", "language", "writer", "plot", "name"):
            if key in extracted:
                self.constraints[key] = extracted[key]

        # Keep the constraint dict from ballooning indefinitely
        allowed_order = ["genre", "plot", "actor", "director", "writer", "language", "year", "name"]
        for key in list(self.constraints.keys()):
            if key not in allowed_order:
                self.constraints.pop(key, None)

    def _format_constraint_summary(self) -> str:
        if not self.constraints:
            return "General movie recommendations"
        parts: List[str] = []
        genre = self.constraints.get("genre")
        if genre:
            parts.append(f"genre: {genre}")
        for key in ("year", "actor", "director", "writer", "language"):
            if value := self.constraints.get(key):
                parts.append(f"{key}: {value}")
        if plot := self.constraints.get("plot"):
            if isinstance(plot, list):
                parts.append(f"plot keywords: {', '.join(plot)}")
            else:
                parts.append(f"plot keyword: {plot}")
        return "; ".join(parts)

    def respond(self, user_message: str) -> str:
        """
        A more robust, single-step response generation.
        1. Extract genre from user message heuristically.
        2. Fetch recommendations based on the extracted genre.
        3. Generate a conversational reply using the recommendations as context.
        """
        # Step 1: Heuristic extraction is faster and good enough for this stage.
        self._update_constraints(user_message)

        # Step 2: Fetch recommendations.
        recs: list[dict] = recommend(self.constraints)
        
        # Prepare a detailed string of recommendations for the prompt and get titles
        rec_details_str = ""
        rec_titles = []
        if recs:
            # Limit to top 3-4 recommendations to keep the prompt focused
            for movie in recs[:4]:
                # Use .get() for safety, in case some keys are missing
                if title := movie.get('name'):
                    rec_titles.append(title)
                    year = movie.get('year', 'N/A')
                    plot = movie.get('plot', 'No plot available.')
                    actors = ", ".join(movie.get('actor', []))
                    rec_details_str += f"- Title: {title} ({year}) | Plot: {plot} | Starring: {actors}\n"
        self.last_rec_titles = rec_titles.copy()

        # Step 3: Generate a high-quality, conversational reply in one shot.
        constraints_text = self._format_constraint_summary()
        reply_prompt = (
            f"You are a helpful and friendly movie recommender. "
            f"The user just said: '{user_message}'. "
            f"User preference cues → {constraints_text}.\n"
            f"Based on those cues, here are some potential recommendations:\n{rec_details_str or '- No precise matches; fall back to broadly liked titles.'}\n"
            f"Politely and conversationally recommend one or two of these movies from this list: {rec_titles}. You can mention the plot or actors to make it more interesting. Keep your response to 1-3 sentences."
        )
        reply = self.call_ollama(reply_prompt)
        return reply
