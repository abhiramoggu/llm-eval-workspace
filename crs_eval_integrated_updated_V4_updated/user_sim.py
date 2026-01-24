# user_sim.py
# User simulator for controlled preference shifts in a closed movie catalog.
#
# Key design goals:
# - Generate natural multi-turn user utterances.
# - Support *dynamic preference focus* (genre-centric, actor-centric, director-centric, etc.).
# - Produce *controlled* focus shifts via an overlapping "bridge" concept from the current context.
# - Emit per-turn metadata (focus field/value, anchor item, shift events) for segment-aware evaluation.

import random
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from config import MODE, USER_MODEL, ERROR_RATE, SHIFT_AFTER_TURNS
from dataset import (
    recommend,
    get_available_genres,
    get_attribute_values,
    MOVIE_DB,
    find_plot_keywords_in_text,
)

# -----------------------------
# Simulator state definitions
# -----------------------------

FOCUS_FIELDS = ["genre", "actor", "director", "writer", "language", "year", "name", "plot"]


@dataclass
class ShiftEvent:
    """Metadata describing a preference-focus shift issued by the simulator."""
    turn: int
    from_focus: str
    to_focus: str
    bridge_field: str
    bridge_value: str
    from_anchor: Optional[str] = None
    to_anchor: Optional[str] = None


class UserSimulator:
    """
    Template-driven simulator with optional LLM phrasing mode.
    The internal state tracks a *preference focus* (field + value) and an anchor movie.
    """

    DETAIL_FOCUSES = ["actors", "themes", "mood"]
    EMOTIONAL_REACTIONS = [
        "That sounds intriguing.",
        "Hmm, I didn’t expect that vibe.",
        "That feels deeper than I thought.",
        "Maybe I’m overthinking it, but it sounds fascinating.",
        "I can’t decide if I’d enjoy it or be scared by it.",
    ]
    CONTRADICTIONS = [
        "Actually, now that I think about it, I might want the opposite.",
        "Maybe I was wrong earlier.",
        "Forget what I said before — that might’ve been too specific.",
        "I’m not sure anymore; perhaps I changed my mind.",
        "It’s weird — I thought I liked that, but maybe not.",
    ]

    ATTRIBUTE_PROMPTS = {
        "year": "Could we look at films from around {value}? {emotion}",
        "actor": "Maybe we could find more movies with {value}. {emotion}",
        "director": "Could we switch to something else by {value}? {emotion}",
        "language": "How about something primarily in {value}? {emotion}",
        "writer": "Do you have recommendations written by {value}? {emotion}",
        "plot": "Maybe explore stories about {value}? {emotion}",
        "name": "Could we talk about {value} specifically? {emotion}",
        "genre": "Could we explore {value} next? {emotion}",
    }

    ACTOR_KEYWORDS = {"actor", "actors", "cast", "starring", "performer", "lead"}
    THEME_KEYWORDS = {"theme", "themes", "story", "plot", "message", "idea", "motif"}
    MOOD_KEYWORDS = {"mood", "tone", "atmosphere", "vibe", "feeling", "emotion"}

    def __init__(self, use_llm_drift: bool = False, seed: Optional[int] = None):
        # Deterministic simulator policy when a seed is provided
        if seed is not None:
            random.seed(seed)

        self.genres = get_available_genres()
        self.attribute_pools: Dict[str, List[str]] = {
            attr: get_attribute_values(attr)
            for attr in ("genre", "year", "actor", "director", "language", "writer", "plot", "name")
        }
        self.attribute_pools["genre"] = sorted(list(self.genres))

        # --- initial focus (genre-centric) ---
        self.true_genre = random.choice(self.genres) if self.genres else "drama"
        requested_genre = self.true_genre

        # optional "wrong start" noise (to test recovery)
        if random.random() < ERROR_RATE and self.genres:
            alternatives = [g for g in self.genres if g != self.true_genre]
            if alternatives:
                requested_genre = random.choice(alternatives)

        self.focus_field: str = "genre"
        self.focus_value: str = requested_genre

        # constraints represent the user's *current active* preferences
        self.current_constraints: Dict[str, object] = {"genre": requested_genre}

        # select an initial anchor item consistent with the focus
        self.anchor_title: Optional[str] = self._pick_anchor_from_constraints(self.current_constraints)

        # stage controls the "flow": initial request -> details -> shift -> details...
        self.stage = "initial"
        self.remaining_details = self._new_detail_sequence()

        # last system state (used to decide what to ask next)
        self.last_system_message = ""
        self.last_recommended_titles: List[str] = []
        self.last_selected_title: Optional[str] = None
        self.system_cues = set()

        # bookkeeping
        self.turn_count = 0
        self.conversation_history: List[Dict[str, str]] = []
        self.use_llm_drift = use_llm_drift

        # shift event tracking (for segment-aware evaluation)
        self.shift_events: List[ShiftEvent] = []
        self._last_user_meta: Dict[str, object] = {}

    # -----------------------------
    # Optional LLM phrasing mode
    # -----------------------------

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
        output, error = process.communicate(input=prompt)
        err_text = (error or "").strip()
        if err_text and any(kw in err_text.lower() for kw in ["error", "failed"]):
            print(f"Ollama Error ({USER_MODEL}): {err_text}")
        return (output or "").strip()

    # -----------------------------
    # Helpers
    # -----------------------------

    def _new_detail_sequence(self) -> List[str]:
        return random.sample(self.DETAIL_FOCUSES, k=len(self.DETAIL_FOCUSES))

    def _reference_title(self) -> str:
        """Pick a movie title to refer to, preferring an anchor or a recent explicit title."""
        if self.last_selected_title:
            return self.last_selected_title
        if self.anchor_title:
            return self.anchor_title
        if self.last_recommended_titles:
            return random.choice(self.last_recommended_titles)
        return "that recommendation"

    def _movie_details(self, title: str) -> dict:
        return MOVIE_DB.get(title, {}) if title else {}

    def _pick_anchor_from_constraints(self, constraints: Dict[str, object]) -> Optional[str]:
        """Select a concrete title consistent with constraints (used to keep shifts grounded)."""
        try:
            recs = recommend(constraints, limit=20)
        except Exception:
            recs = []
        titles = [r.get("name") for r in recs if r.get("name")]
        # avoid picking the same anchor if possible
        prev_anchor = getattr(self, "anchor_title", None)
        if titles and prev_anchor in titles and len(titles) > 1:
            titles = [t for t in titles if t != self.anchor_title]
        return random.choice(titles) if titles else None

    def _choose_bridge_from_anchor(self) -> Tuple[str, str]:
        """
        Choose an overlapping "bridge" concept from the current anchor context.
        This is what makes topic/focus shifts coherent (not arbitrary).
        """
        title = self._reference_title()
        details = self._movie_details(title)

        candidates: List[Tuple[str, str]] = []

        # Prefer structured fields first
        for field in ("actor", "director", "writer", "language"):
            vals = details.get(field) or []
            for v in vals:
                if v:
                    candidates.append((field, str(v)))

        # Plot keywords as weaker bridges
        plot = details.get("plot", "")
        for kw in find_plot_keywords_in_text(plot, limit=6):
            candidates.append(("plot", kw))

        # Fall back to the active focus itself
        if self.focus_field and self.focus_value:
            candidates.append((self.focus_field, str(self.focus_value)))

        return random.choice(candidates) if candidates else ("genre", str(self.focus_value))

    def _maybe_forget_old_constraints(self, keep: List[str], drop_prob: float = 0.35):
        """Occasionally drop constraints to mimic human indecision (but keep core focus + bridge)."""
        keys = [k for k in list(self.current_constraints.keys()) if k not in keep]
        if keys and random.random() < drop_prob:
            drop_key = random.choice(keys)
            self.current_constraints.pop(drop_key, None)

    def _compose_detail_question(self, focus: str) -> str:
        """Generate curiosity about the current anchor."""
        title = self._reference_title()
        emotion = random.choice(self.EMOTIONAL_REACTIONS)
        if focus == "actors":
            return f"I like the sound of {title}. Who stars in it? {emotion}"
        if focus == "themes":
            return f"What kind of story does {title} tell? {emotion}"
        return f"What's the mood of {title}? {emotion}"

    def _extract_titles(self, text: str) -> List[str]:
        lower = (text or "").lower()
        found = []
        for title in MOVIE_DB.keys():
            if title and title.lower() in lower:
                found.append(title)
        if found:
            self.last_selected_title = random.choice(found)
        return found

    def _reflect_or_contradict(self) -> str:
        return random.choice(self.CONTRADICTIONS)

    # -----------------------------
    # Focus shifting
    # -----------------------------

    def _select_next_focus_field(self, current: str) -> str:
        candidates = [f for f in FOCUS_FIELDS if f != current]
        random.shuffle(candidates)
        # bias slightly towards structured facets
        for fav in ("actor", "director", "genre", "writer", "language", "name", "plot", "year"):
            if fav in candidates and random.random() < 0.55:
                return fav
        return candidates[0] if candidates else current

    def _compose_focus_shift_message(self) -> Tuple[str, Optional[ShiftEvent]]:
        """
        Issue a coherent shift:
        - pick a bridge concept from anchor
        - change the focus field (e.g., genre -> actor)
        - keep the bridge constraint so the shift is explainable
        """
        from_focus = self.focus_field
        from_anchor = self.anchor_title

        bridge_field, bridge_value = self._choose_bridge_from_anchor()
        to_focus = self._select_next_focus_field(from_focus)

        emotion = random.choice(self.EMOTIONAL_REACTIONS)
        reference = self._reference_title()

        # Decide a target focus value
        if to_focus == "genre":
            # choose a different genre, optionally keeping bridge constraint
            possible = [g for g in self.genres if g != str(self.current_constraints.get("genre", ""))]
            to_value = random.choice(possible) if possible else str(self.current_constraints.get("genre", "drama"))
            msg = (
                f"Thanks for all the detail on {reference}. "
                f"I want to try something more {to_value}—"
            )
            if bridge_field in ("actor", "director", "writer", "language"):
                msg += f"still with {bridge_value} if possible. {emotion}"
                self.current_constraints[bridge_field] = bridge_value
            else:
                msg += f"{emotion}"
            self.current_constraints["genre"] = to_value

        elif to_focus in ("actor", "director", "writer", "language", "year"):
            # use bridge if it matches; else sample from pool
            if bridge_field == to_focus:
                to_value = bridge_value
            else:
                pool = self.attribute_pools.get(to_focus) or []
                to_value = random.choice(pool) if pool else bridge_value
            template = self.ATTRIBUTE_PROMPTS.get(to_focus, "Could we focus on {value}? {emotion}")
            msg = template.format(value=to_value, emotion=emotion)
            msg = f"{msg} (I’m thinking of {reference}.)"
            self.current_constraints[to_focus] = to_value
            # keep bridge constraint if it is informative and different
            if bridge_field != to_focus and bridge_field in ("actor", "director", "writer", "language", "plot"):
                self.current_constraints[bridge_field] = bridge_value

        elif to_focus == "plot":
            # prefer plot keyword bridge
            to_value = bridge_value if bridge_field == "plot" else bridge_value
            template = self.ATTRIBUTE_PROMPTS["plot"]
            msg = template.format(value=to_value, emotion=emotion)
            msg = f"{msg} Maybe something that connects to {reference}."
            self.current_constraints["plot"] = [to_value] if isinstance(self.current_constraints.get("plot"), list) else to_value
            if bridge_field in ("actor", "director", "writer", "language"):
                self.current_constraints[bridge_field] = bridge_value

        else:  # to_focus == "name"
            # shift to a *different* concrete title related by the bridge
            candidates = recommend({bridge_field: bridge_value}, limit=25) if bridge_field else []
            titles = [c.get("name") for c in candidates if c.get("name")]
            titles = [t for t in titles if t != self.anchor_title]
            to_value = random.choice(titles) if titles else (self.anchor_title or reference)
            msg = f"Actually, could we talk about {to_value} specifically? {emotion}"
            # focus on a concrete entity
            self.current_constraints["name"] = to_value
            # keep the bridge constraint for explainability
            if bridge_field and bridge_value:
                self.current_constraints[bridge_field] = bridge_value

        # Update focus and anchor
        self.focus_field = to_focus
        self.focus_value = str(self.current_constraints.get(to_focus, "")) if to_focus in self.current_constraints else ""
        self._maybe_forget_old_constraints(keep=[self.focus_field, "genre", bridge_field], drop_prob=0.4)

        # Select a new anchor title consistent with the new constraints (if possible)
        self.anchor_title = self._pick_anchor_from_constraints(self.current_constraints)

        event = ShiftEvent(
            turn=self.turn_count,
            from_focus=from_focus,
            to_focus=to_focus,
            bridge_field=bridge_field,
            bridge_value=str(bridge_value),
            from_anchor=from_anchor,
            to_anchor=self.anchor_title,
        )
        self.shift_events.append(event)
        self.remaining_details = self._new_detail_sequence()
        return msg, event

    # -----------------------------
    # Public interface used by simulate.py
    # -----------------------------

    def get_last_meta(self) -> Dict[str, object]:
        """Return the metadata associated with the *last* generated user message."""
        return dict(self._last_user_meta)

    def get_message(self) -> str:
        """Generate the next user utterance according to current policy."""
        self.turn_count += 1

        # --- Optional LLM-driven drift mode (phrasing only) ---
        if self.use_llm_drift:
            convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in self.conversation_history[-6:]])
            drift_prompt = f"""
Continue this dialogue as the USER. Keep it realistic and coherent.

{convo_text}

Rules:
- Keep preferences grounded in movies/actors/directors/genres/language/year/plot cues.
- Occasionally refine constraints or shift focus using something overlapping from context.
- 1–3 sentences only; sound human.
"""
            msg = self.call_ollama(drift_prompt)
            self._last_user_meta = {
                "turn": self.turn_count,
                "stage": self.stage,
                "focus_field": self.focus_field,
                "focus_value": self.focus_value,
                "anchor_title": self.anchor_title,
                "constraints": dict(self.current_constraints),
                "shift_event": False,
            }
            self.conversation_history.append({"speaker": "USER", "text": msg})
            return msg

        # --- Template-driven behavior (rule-based) ---
        shift_event_obj: Optional[ShiftEvent] = None

        if self.stage == "initial":
            msg = f"I'm in the mood for a {self.current_constraints.get('genre', self.focus_value)} movie. Could you recommend a few?"
            self.stage = "details"

        elif self.stage == "details":
            # sometimes reflect mid-conversation
            if random.random() < 0.15:
                msg = self._reflect_or_contradict()
            # ensure at least one shift by the configured turn
            elif (not self.shift_events) and (self.turn_count >= max(1, SHIFT_AFTER_TURNS)):
                msg, shift_event_obj = self._compose_focus_shift_message()
                self.stage = "waiting_system"
            # shift after exhausting detail questions OR occasionally earlier
            elif (not self.remaining_details) or (random.random() < 0.18):
                msg, shift_event_obj = self._compose_focus_shift_message()
                self.stage = "waiting_system"
            else:
                focus = self.remaining_details.pop(0)
                msg = self._compose_detail_question(focus)

        else:  # waiting_system
            msg = (
                "Just checking—do you have more recommendations like that?"
                if self.last_system_message.strip()
                else "Could you share a bit more detail when you get a chance?"
            )
            self.stage = "details"

        self._last_user_meta = {
            "turn": self.turn_count,
            "stage": self.stage,
            "focus_field": self.focus_field,
            "focus_value": self.focus_value,
            "anchor_title": self.anchor_title,
            "constraints": dict(self.current_constraints),
            "shift_event": bool(shift_event_obj),
            "shift_event_obj": asdict(shift_event_obj) if shift_event_obj else None,
        }
        self.conversation_history.append({"speaker": "USER", "text": msg})
        return msg

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
