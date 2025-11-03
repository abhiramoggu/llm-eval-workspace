# user_sim.py
# Extended user simulator with hybrid topic drift, emotional recall, and nuanced genre blending.

import random
import subprocess
from typing import List

from config import MODE, USER_MODEL, ERROR_RATE
from dataset import recommend


def _available_genres() -> List[str]:
    """Extract the genre tuple embedded in dataset.recommend()."""
    consts = recommend.__code__.co_consts
    for const in reversed(consts):
        if isinstance(const, tuple) and const and all(isinstance(x, str) for x in const):
            return list(const)
    return ["romance", "horror", "sci-fi", "action", "comedy", "drama", "thriller", "fantasy", "animation", "crime", "mystery", "adventure"]


class UserSimulator:
    DETAIL_FOCUSES = ["actors", "themes", "mood"]
    EMOTIONAL_REACTIONS = [
        "That sounds intriguing.",
        "Hmm, I didn’t expect that vibe.",
        "That feels deeper than I thought.",
        "Maybe I’m overthinking it, but it sounds fascinating.",
        "I can’t decide if I’d enjoy it or be scared by it."
    ]
    CONTRADICTIONS = [
        "Actually, now that I think about it, I might want the opposite.",
        "Maybe I was wrong earlier.",
        "Forget what I said before — that might’ve been too specific.",
        "I’m not sure anymore; perhaps I changed my mind.",
        "It’s weird — I thought I liked that, but maybe not."
    ]

    ACTOR_KEYWORDS = {"actor", "actors", "cast", "starring", "performer", "lead"}
    THEME_KEYWORDS = {"theme", "themes", "story", "plot", "message", "idea", "motif"}
    MOOD_KEYWORDS = {"mood", "tone", "atmosphere", "vibe", "feeling", "emotion"}

    def __init__(self, use_llm_drift: bool = False):
        self.genres = _available_genres()
        self.catalog = {
            genre: [movie["title"] for movie in recommend({"genre": genre})]
            for genre in self.genres
        }
        self.all_titles = [title for titles in self.catalog.values() for title in titles]

        self.true_genre = random.choice(self.genres)
        self.current_topic = self.true_genre
        self.visited_genres = {self.current_topic}

        self.stage = "initial"
        self.remaining_details = self._new_detail_sequence()
        self.last_system_message = ""
        self.last_recommended_titles: List[str] = []
        self.last_selected_title = None
        self.system_cues = set()
        self.next_genre_candidate = None

        self.turn_count = 0
        self.conversation_history = []
        self.use_llm_drift = use_llm_drift

    def _new_detail_sequence(self) -> List[str]:
        return random.sample(self.DETAIL_FOCUSES, k=len(self.DETAIL_FOCUSES))

    # ---------- Optional LLM drift ----------
    def call_ollama(self, prompt: str) -> str:
        """LLM-backed phrasing (used in drift mode)."""
        if MODE == "mock":
            return "I'd love to hear more about those choices."
        process = subprocess.Popen(
            ["ollama", "run", USER_MODEL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, _ = process.communicate(input=prompt)
        return output.strip()

    # ---------- Message helpers ----------

    def _choose_detail_focus(self) -> str:
        """Select which detail (actors/themes/mood) to ask about next."""
        for focus in ("actors", "themes", "mood"):
            if focus in self.system_cues and focus in self.remaining_details:
                self.remaining_details.remove(focus)
                self.system_cues.discard(focus)
                return focus
        return self.remaining_details.pop(0)

    def _reference_title(self) -> str:
        """Pick a movie to refer back to."""
        if self.last_selected_title:
            return self.last_selected_title
        titles = self.catalog.get(self.current_topic, [])
        if titles:
            return random.choice(titles)
        return "that recommendation"

    def _compose_detail_question(self, focus: str) -> str:
        """Generate human-like curiosity about a movie."""
        title = self._reference_title()
        emotion = random.choice(self.EMOTIONAL_REACTIONS)
        if focus == "actors":
            return f"I like the sound of {title}. Who stars in it? {emotion}"
        if focus == "themes":
            return f"What kind of story does {title} tell? {emotion}"
        return f"What's the mood of {title}? {emotion}"

    def _random_new_genre(self) -> str:
        """Choose a new genre, sometimes blending with the old."""
        candidates = [g for g in self.genres if g != self.current_topic]
        random.shuffle(candidates)
        if random.random() < 0.4:
            # blend genres occasionally
            blend = f"{self.current_topic} and {random.choice(candidates)}"
            return blend
        for candidate in candidates:
            if candidate not in self.visited_genres or random.random() < 0.25:
                return candidate
        return candidates[0] if candidates else self.current_topic

    def _compose_genre_shift_message(self) -> str:
        """Transition naturally to a new or blended interest."""
        target = self.next_genre_candidate or self._random_new_genre()
        reference = self._reference_title()
        blend_phrase = random.choice([
            "but maybe with a softer touch",
            "though something emotional wouldn’t hurt",
            "with a mix of excitement and warmth",
            "something that surprises me emotionally"
        ])
        message = (
            f"Thanks for all the detail on {reference}. "
            f"Could we explore {target} next, {blend_phrase}?"
        )
        self.current_topic = target
        self.visited_genres.add(target)
        self.next_genre_candidate = None
        self.remaining_details = self._new_detail_sequence()
        return message

    def _reflect_or_contradict(self) -> str:
        """Occasional reflective or contradictory statement."""
        return random.choice(self.CONTRADICTIONS)

    # ---------- Conversation loop ----------

    def get_message(self):
        """Generate the next user utterance according to current policy."""
        self.turn_count += 1

        # --- Optional LLM-driven drift mode ---
        if self.use_llm_drift:
            convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in self.conversation_history[-6:]])
            drift_prompt = f"""
Continue this dialogue as the USER, following realistic conversation flow.

{convo_text}

Rules:
- Shift topics naturally or blend interests.
- Occasionally contradict yourself or recall earlier parts.
- 1–3 sentences only; sound human and reflective.
"""
            msg = self.call_ollama(drift_prompt)
            self.conversation_history.append({"speaker": "USER", "text": msg})
            return msg

        # --- Template-driven behavior (rule-based) ---
        if self.stage == "initial":
            requested_genre = self.current_topic
            if random.random() < ERROR_RATE:
                alternatives = [g for g in self.genres if g != self.true_genre]
                if alternatives:
                    requested_genre = random.choice(alternatives)
                    self.current_topic = requested_genre
            msg = f"I'm in the mood for a {requested_genre} movie. Could you recommend a few?"
            self.stage = "details"

        elif self.stage == "details":
            # sometimes reflect mid-conversation
            if random.random() < 0.15:
                msg = self._reflect_or_contradict()
            elif not self.remaining_details:
                self.stage = "awaiting_shift"
                msg = self._compose_genre_shift_message()
                self.stage = "waiting_system"
            else:
                focus = self._choose_detail_focus()
                msg = self._compose_detail_question(focus)
                if not self.remaining_details:
                    self.stage = "awaiting_shift"

        elif self.stage == "awaiting_shift":
            msg = self._compose_genre_shift_message()
            self.stage = "waiting_system"

        else:  # waiting_system
            msg = (
                "Just checking—do you have more recommendations like that?"
                if self.last_system_message.strip()
                else "Could you share a bit more detail when you get a chance?"
            )

        self.conversation_history.append({"speaker": "USER", "text": msg})
        return msg

    def _extract_titles(self, text: str) -> List[str]:
        lower = text.lower()
        found = []
        for genre, titles in self.catalog.items():
            for title in titles:
                if title.lower() in lower:
                    found.append((genre, title))
        if found:
            preferred = [title for genre, title in found if genre == self.current_topic]
            if preferred:
                self.last_selected_title = random.choice(preferred)
            else:
                self.last_selected_title = random.choice([title for _, title in found])
            return [title for _, title in found]
        return []

    def record_system_message(self, message: str):
        """Store the CRS response and mine it for cues before the next turn."""
        self.conversation_history.append({"speaker": "SYSTEM", "text": message})
        self.last_system_message = message or ""
        lowered = self.last_system_message.lower()

        self.last_recommended_titles = self._extract_titles(self.last_system_message)

        self.system_cues = set()
        if any(word in lowered for word in self.ACTOR_KEYWORDS):
            self.system_cues.add("actors")
        if any(word in lowered for word in self.THEME_KEYWORDS):
            self.system_cues.add("themes")
        if any(word in lowered for word in self.MOOD_KEYWORDS):
            self.system_cues.add("mood")

        genre_mentions = [
            genre for genre in self.genres if genre != self.current_topic and genre in lowered
        ]
        if genre_mentions:
            self.next_genre_candidate = random.choice(genre_mentions)

        if self.stage == "waiting_system":
            self.stage = "details"
