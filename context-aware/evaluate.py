"""
Hybrid evaluation for CRS:
- Topic extraction (LLM or heuristic)
- Topic segmentation & dynamic intent tracking
- Recovery detection (rate, delay)
- Interference (old topic leakage)
- Cross-coherence (User↔CRS)
- Context retention (CRS internal)
- CAS: Composite Context Adaptation Score
- Optional LLM-as-judge (qualitative)
"""

import json
import re
import subprocess
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional

from config import (
    MODE,
    JUDGE_MODEL,
    TOPIC_EXTRACTOR_MODE,
    TOPIC_EXTRACTOR_MODEL,
    SIM_TOPIC_SHIFT,
    ALIGNMENT_THRESHOLD,
    TOPIC_JACCARD_SHIFT,
    CAS_WEIGHTS,
    EMBEDDING_MODEL_NAME,
    VERBOSE_EVAL,
)

TOPIC_CACHE: Dict[str, List[str]] = {}
SIM_CACHE: Dict[Tuple[str, str], float] = {}
_EMBEDDING_MODEL: Optional[Any] = None


def ensure_embedding_model():
    """Lazy-loads the sentence-transformer model to avoid loading it on every import."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except ImportError:
            print("Warning: sentence-transformers is not installed. Similarity metrics will be degraded.")
            _EMBEDDING_MODEL = "unavailable"


def call_ollama_json(prompt: str, model_name: str) -> Optional[dict]:
    process = subprocess.Popen(
        ["ollama", "run", model_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output, error = process.communicate(input=prompt)
    if error:
        print("Error:", error)
    if not output:
        return None
    try:
        return json.loads(output.strip())
    except Exception:
        return None


# ---------------- Topic extraction ----------------

_STOPWORDS = set("""
a an the and or of to in on for with at from by this that these those about as into like over after before between during
is are was were be been being have has had do does did can could should would will may might must not no nor very more most
""".split())

DOMAIN_HINTS = {
    "romance", "horror", "sci-fi", "science fiction", "drama", "comedy", "thriller",
    "actor", "actors", "cast", "director", "plot", "ending", "mood", "emotional",
    "gore", "space", "aliens", "ghost", "jump scare", "recommendation", "movie", "film"
}


def _filter_topics(topics: List[str]) -> List[str]:
    """Filter out irrelevant, generic, or duplicate topics."""
    if not topics:
        return []
    irrelevant = {
        "recommendation", "conversation", "system", "response", "chat", "user", "talk",
        "suggestion", "question", "answer", "film", "movie", "thing", "stuff"
    }
    filtered = []
    for t in topics:
        t = t.strip().lower()
        if not t or len(t) < 2:
            continue
        if t in irrelevant:
            continue
        if any(k in t for k in DOMAIN_HINTS) or re.search(
            r"(love|fear|sad|happy|romantic|tense|dark|futur|drama|emot|space|ghost|sci)", t
        ):
            filtered.append(t)
    return list(dict.fromkeys(filtered))[:5]


def extract_topics_heuristic(text: str, top_k: int = 5) -> List[str]:
    """Enhanced keyword-based heuristic with filtering."""
    text_l = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    tokens = [t for t in text_l.split() if t and t not in _STOPWORDS]
    scored = {}
    for t in tokens:
        score = 1
        if t in DOMAIN_HINTS:
            score += 2
        if re.search(r"(love|fear|sad|happy|romantic|tense|dark|space|sci|ghost|drama|emot)", t):
            score += 2
        scored[t] = scored.get(t, 0) + score
    topics = [w for w, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)]
    return _filter_topics(topics[:top_k])


def call_ollama(model_name: str, prompt: str) -> str:
    try:
        proc = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = proc.communicate(input=prompt)
    except FileNotFoundError:
        return ""
    if err:
        pass
    return (out or "").strip()


def extract_topics_llm(text: str, model: str = TOPIC_EXTRACTOR_MODEL, top_k: int = 5) -> List[str]:
    """LLM-based extractor with topic filtering."""
    if text in TOPIC_CACHE:
        return TOPIC_CACHE[text][:top_k]

    prompt = f"""
Extract the 3–5 most meaningful *thematic* topics from this message in the context of movie or content recommendations.
Focus on genres, emotions, tones, or concepts (e.g., 'romantic', 'space', 'fear', 'nostalgia').
Return a JSON array only, no commentary.

Text: {text}
"""
    topics = []
    if MODE == "ollama":
        payload = call_ollama_json(prompt, model)
        if isinstance(payload, list):
            topics = [str(x).lower() for x in payload]
    if not topics:
        topics = extract_topics_heuristic(text, top_k)
    filtered = _filter_topics(topics)
    TOPIC_CACHE[text] = filtered
    return filtered[:top_k]


def extract_topics(text: str) -> List[str]:
    if TOPIC_EXTRACTOR_MODE == "llm":
        return extract_topics_llm(text)
    return extract_topics_heuristic(text)


def topics_to_text(topics: List[str]) -> str:
    return ", ".join(topics) if topics else ""


def embedding_similarity(text_a: str, text_b: str) -> float:
    """Calculate semantic similarity using a sentence-transformer model."""
    ensure_embedding_model()
    key = tuple(sorted((text_a, text_b)))
    if key in SIM_CACHE:
        return SIM_CACHE[key]

    if _EMBEDDING_MODEL == "unavailable" or not hasattr(_EMBEDDING_MODEL, 'encode'):
        sim = SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
        SIM_CACHE[key] = sim
        return sim

    from sentence_transformers import util
    embeddings = _EMBEDDING_MODEL.encode([text_a, text_b], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    SIM_CACHE[key] = sim
    return sim


# ---------------- Conversation utilities ----------------

def conversation_pairs(conversation: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs = []
    for i in range(0, len(conversation) - 1, 2):
        if conversation[i]["speaker"] == "USER" and conversation[i + 1]["speaker"] == "SYSTEM":
            pairs.append((conversation[i]["text"], conversation[i + 1]["text"]))
    return pairs


def avg(lst: List[float]) -> Optional[float]:
    lst = [x for x in lst if x is not None]
    if not lst:
        return None
    return sum(lst) / len(lst)


def jaccard_overlap(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------- Segmentation ----------------

def detect_user_segments(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Segment USER utterances into contiguous topic segments using embedding similarity."""
    user_idxs = [i for i, t in enumerate(conversation) if t["speaker"] == "USER"]
    if not user_idxs:
        return []

    segments = []
    current = {"start_idx": None, "end_idx": None, "texts": []}
    prev_text = ""
    prev_topics: List[str] = []

    for i, t in enumerate(conversation):
        if t["speaker"] != "USER":
            continue
        txt = t["text"]
        if current["start_idx"] is None:
            current = {"start_idx": i, "end_idx": i, "texts": [txt]}
            prev_topics = extract_topics(txt)
            prev_text = txt
        else:
            base = prev_text if prev_text else " ".join(prev_topics)
            sim = embedding_similarity(base, txt)
            curr_topics = extract_topics(txt)
            topic_jaccard = jaccard_overlap(prev_topics, curr_topics)
            shift_detected = (sim < SIM_TOPIC_SHIFT) or (topic_jaccard < TOPIC_JACCARD_SHIFT)
            if shift_detected:
                seg_txt = " ".join(current["texts"])
                segments.append({
                    "start_idx": current["start_idx"],
                    "end_idx": current["end_idx"],
                    "topics": _filter_topics(extract_topics(seg_txt)),
                    "repr_text": seg_txt
                })
                current = {"start_idx": i, "end_idx": i, "texts": [txt]}
            else:
                current["end_idx"] = i
                current["texts"].append(txt)
            prev_topics = curr_topics
            prev_text = txt

    if current["start_idx"] is not None:
        seg_txt = " ".join(current["texts"])
        segments.append({
            "start_idx": current["start_idx"],
            "end_idx": current["end_idx"],
            "topics": _filter_topics(extract_topics(seg_txt)),
            "repr_text": seg_txt
        })

    return segments


# ---------------- Metrics ----------------

def find_first_system_after(conversation, idx_user_end) -> Optional[int]:
    for j in range(idx_user_end + 1, len(conversation)):
        if conversation[j]["speaker"] == "SYSTEM":
            return j
    return None


def system_alignment_with_topics(system_text: str, user_topics: List[str]) -> Optional[float]:
    if not system_text.strip() or not user_topics:
        return 0.0
    topics_text = " ".join(user_topics)
    return embedding_similarity(system_text, topics_text)


def cross_coherence(conversation: List[Dict[str, Any]]) -> Optional[float]:
    sims = []
    for i in range(0, len(conversation) - 1, 2):
        a, b = conversation[i], conversation[i + 1]
        if a["speaker"] == "USER" and b["speaker"] == "SYSTEM":
            sims.append(embedding_similarity(a["text"], b["text"]))
    return sum(sims) / len(sims) if sims else 0.0


def context_retention(conversation: List[Dict[str, Any]]) -> Optional[float]:
    sims = []
    prev_sys = None
    for t in conversation:
        if t["speaker"] != "SYSTEM":
            continue
        if prev_sys is not None:
            sims.append(embedding_similarity(prev_sys["text"], t["text"]))
        prev_sys = t
    return sum(sims) / len(sims) if sims else 0.0


def compute_recovery_and_interference(conversation: List[Dict[str, Any]], user_segments: List[Dict[str, Any]]):
    details = []
    total_shifts = max(0, len(user_segments) - 1)
    recovered_count = 0
    delays = []
    interferences = []

    if total_shifts == 0:
        return {
            "topic_recovery_rate": 0.0,
            "avg_recovery_delay": 0.0,
            "topic_interference": 0.0,
            "per_shift": []
        }

    for k in range(total_shifts):
        prev_seg = user_segments[k]
        new_seg = user_segments[k + 1]

        new_topics = new_seg["topics"]
        old_topics = prev_seg["topics"]
        start_check_idx = find_first_system_after(conversation, new_seg["start_idx"])

        recovered = False
        delay_sys_turns = 0
        leakage_hits = 0
        sys_seen = 0

        if start_check_idx is not None:
            for j in range(start_check_idx, len(conversation)):
                t = conversation[j]
                if t["speaker"] != "SYSTEM":
                    continue
                sys_seen += 1

                sys_topics = _filter_topics(extract_topics(t["text"]))
                important_new = _filter_topics(new_topics)
                important_old = _filter_topics(old_topics)

                if not sys_topics:
                    continue

                if important_new:
                    align = jaccard_overlap(sys_topics, important_new)
                    if align < 0.3:
                        align = embedding_similarity(" ".join(sys_topics), " ".join(important_new))
                else:
                    align = 0.0

                leak = 0.0
                if important_old:
                    leak = jaccard_overlap(sys_topics, important_old)
                    if leak < 0.3:
                        leak = embedding_similarity(" ".join(sys_topics), " ".join(important_old))

                if leak >= ALIGNMENT_THRESHOLD:
                    leakage_hits += 1

                if align >= ALIGNMENT_THRESHOLD:
                    recovered = True
                    delay_sys_turns = sys_seen
                    break

        if recovered:
            recovered_count += 1
            delays.append(delay_sys_turns)
        else:
            delays.append(None)

        denom = max(1, (delay_sys_turns if recovered else min(sys_seen, 4)))
        interference = leakage_hits / denom if denom else 0.0
        interferences.append(interference)

        details.append({
            "shift_index": k,
            "from_topics": old_topics,
            "to_topics": new_topics,
            "recovered": recovered,
            "recovery_delay_sys_turns": delay_sys_turns if recovered else None,
            "interference": interference
        })

    trr = recovered_count / total_shifts if total_shifts > 0 else 0.0
    valid_delays = [d for d in delays if d is not None]
    avg_delay = sum(valid_delays) / len(valid_delays) if valid_delays else 0.0
    avg_interference = sum(interferences) / len(interferences) if interferences else 0.0

    return {
        "topic_recovery_rate": trr,
        "avg_recovery_delay": avg_delay,
        "topic_interference": avg_interference,
        "per_shift": details
    }


def normalize(value: Optional[float], lo: float, hi: float, invert: bool = False) -> Optional[float]:
    if value is None:
        return None
    if hi == lo:
        return 0.0
    x = (value - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    return 1.0 - x if invert else x


def compute_cas(metrics: Dict[str, Optional[float]]) -> Optional[float]:
    trr = metrics.get("topic_recovery_rate")
    rd = normalize(metrics.get("avg_recovery_delay"), 1, 6, invert=True)
    ti = normalize(metrics.get("topic_interference"), 0, 1, invert=True)
    cc = metrics.get("cross_coherence")
    cr = metrics.get("context_retention")

    parts = {
        "topic_recovery_rate": trr,
        "avg_recovery_delay": rd,
        "topic_interference": ti,
        "cross_coherence": cc,
        "context_retention": cr
    }

    wsum = sum(CAS_WEIGHTS.values())
    weights = {k: v / wsum for k, v in CAS_WEIGHTS.items()}

    score, denom = 0.0, 0.0
    for k, w in weights.items():
        v = parts.get(k)
        if v is not None:
            score += w * v
            denom += w
    if denom == 0:
        return None
    return score / denom


def compute_turn_alignments(conversation: List[Dict[str, Any]], user_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For each system turn, compute its alignment with the current user topic segment."""
    if not user_segments:
        return []

    alignments = []
    current_segment_idx = 0
    for turn_idx, turn in enumerate(conversation):
        if turn["speaker"] != "SYSTEM":
            continue

        # Find which user segment this system turn belongs to
        while (current_segment_idx + 1 < len(user_segments) and
               turn_idx > user_segments[current_segment_idx]["end_idx"]):
            current_segment_idx += 1

        user_segment_text = user_segments[current_segment_idx]["repr_text"]
        alignment_score = embedding_similarity(turn["text"], user_segment_text)
        alignments.append({"turn_idx": turn_idx, "alignment": alignment_score})

    return alignments


def llm_judge(conversation: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """LLM-as-judge for qualitative metrics."""
    convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in conversation])
    prompt = f"""
You are a fair and impartial evaluator for conversational AI.
Below is a conversation between a USER and a SYSTEM. The user's goal is to get movie recommendations.
The user may change their mind or correct themselves. Your task is to evaluate how well the SYSTEM handled the conversation.

Conversation:
{convo_text}

Please rate the SYSTEM's performance on the following metrics on a scale from 0 (very poor) to 5 (excellent).
Provide your response as a JSON object only, with no other text.

- clarity: Was the system's language clear and easy to understand?
- politeness: Was the system polite and respectful?
- recovery: When the user changed topics or corrected a mistake, how well did the system adapt?
- context_memory: How well did the system remember earlier parts of the conversation?
- engagement: Was the system engaging and conversational, or just robotic?

JSON response format:
{{"clarity": <0-5>, "politeness": <0-5>, "recovery": <0-5>, "context_memory": <0-5>, "engagement": <0-5>}}
"""

    if MODE == "mock":
        return {"clarity": 4, "politeness": 5, "recovery": 3, "context_memory": 4, "engagement": 4}

    payload = call_ollama_json(prompt, JUDGE_MODEL)

    if payload and isinstance(payload, dict):
        # Ensure all expected keys are present, defaulting to None
        expected_keys = ["clarity", "politeness", "recovery", "context_memory", "engagement"]
        return {key: payload.get(key) for key in expected_keys}

    return {
        "clarity": None,
        "politeness": None,
        "recovery": None,
        "context_memory": None,
        "engagement": None,
    }


# ---------------- Public API ----------------

def evaluate(conversation: List[Dict[str, Any]], true_genre: Optional[str] = None) -> Dict[str, Any]:
    if VERBOSE_EVAL:
        print(f"\n[EVAL] Starting evaluation: {len(conversation)} turns, true_genre={true_genre}")

    user_segments = detect_user_segments(conversation)
    cc = cross_coherence(conversation)
    cr = context_retention(conversation)
    ri = compute_recovery_and_interference(conversation, user_segments)
    turn_alignments = compute_turn_alignments(conversation, user_segments)

    metrics = {
        "topic_recovery_rate": ri["topic_recovery_rate"],
        "avg_recovery_delay": ri["avg_recovery_delay"],
        "topic_interference": ri["topic_interference"],
        "cross_coherence": cc,
        "context_retention": cr,
    }

    cas = compute_cas(metrics)
    metrics["context_adaptation_score"] = cas
    metrics.update({
        "true_genre": true_genre,
        "num_topic_shifts": max(0, len(user_segments) - 1),
    })

    result = {
        **metrics,
        "detail": {
            "user_segments": user_segments,
            "per_shift": ri["per_shift"],
            "turn_alignments": turn_alignments
        }
    }
    return result
