# evaluate.py
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
from typing import List, Dict, Any, Tuple, Optional

from config import (
    MODE, JUDGE_MODEL, TOPIC_EXTRACTOR_MODE, TOPIC_EXTRACTOR_MODEL,
    SIM_TOPIC_SHIFT, ALIGNMENT_THRESHOLD, CAS_WEIGHTS, EMBEDDING_MODEL_NAME
)

# -------- Optional embeddings (lazy-load) --------
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

_embedding_model = None
_embedding_load_failed = False


def ensure_embedding_model():
    """Lazily load the embedding model; returns None if unavailable."""
    global _embedding_model, _embedding_load_failed
    if _embedding_model is not None:
        return _embedding_model
    if SentenceTransformer is None or _embedding_load_failed:
        return None
    try:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        print(f"Warning: embedding model unavailable ({exc}). Skipping semantic metrics.")
        _embedding_load_failed = True
        return None
    return _embedding_model


def embed(text: str):
    model = ensure_embedding_model()
    if model is None:
        return None
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)


def cos_sim(a, b) -> Optional[float]:
    if a is None or b is None or util is None:
        return None
    return float(util.cos_sim(a, b))


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


def extract_topics_heuristic(text: str, top_k: int = 5) -> List[str]:
    """Very light, dependency-free keyword heuristic."""
    # Lowercase, keep words and phrases
    text_l = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    tokens = [t for t in text_l.split() if t and t not in _STOPWORDS]
    # Keep domain hints and frequent tokens
    scored = {}
    for t in tokens:
        score = 1
        if t in DOMAIN_HINTS:
            score += 2
        scored[t] = scored.get(t, 0) + score
    # Sort & slice
    topics = [w for w, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)]
    return topics[:top_k]


def call_ollama(model_name: str, prompt: str) -> str:
    proc = subprocess.Popen(
        ["ollama", "run", model_name],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate(input=prompt)
    if err:
        # Ollama often writes model load info to stderr; ignore non-fatal
        pass
    return (out or "").strip()


def extract_topics_llm(text: str, model: str = TOPIC_EXTRACTOR_MODEL, top_k: int = 5) -> List[str]:
    """LLM-based topic extractor; expects JSON list output. Falls back to heuristic."""
    prompt = f"""
Extract the 3-5 most important topics/entities from the following user utterance for a movie recommendation context.
Return a JSON array of short strings, no commentary.

Text: {text}
"""
    if MODE == "ollama":
        raw = call_ollama(model, prompt)
        # Try parse JSON list
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                topics = [str(x).lower() for x in parsed][:top_k]
                return topics
        except Exception:
            pass
    # Fallback
    return extract_topics_heuristic(text, top_k)


def extract_topics(text: str) -> List[str]:
    if TOPIC_EXTRACTOR_MODE == "llm":
        return extract_topics_llm(text)
    return extract_topics_heuristic(text)


def topics_to_text(topics: List[str]) -> str:
    return ", ".join(topics) if topics else ""


# ---------------- Conversation utilities ----------------

def conversation_pairs(conversation: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Return [(user_text, system_text), ...] aligned by adjacency; skips if missing."""
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


# ---------------- Segmentation & tracking ----------------

def detect_user_segments(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Segment USER utterances into contiguous topic segments using embedding similarity.
    Returns a list of segments: [{start_idx, end_idx, topics, repr_text}, ...]
    """
    # If embeddings absent, single segment
    if embed("x") is None or util is None:
        user_idxs = [i for i, t in enumerate(conversation) if t["speaker"] == "USER"]
        if not user_idxs:
            return []
        texts = [conversation[i]["text"] for i in user_idxs]
        return [{"start_idx": user_idxs[0], "end_idx": user_idxs[-1], "topics": extract_topics(" ".join(texts)), "repr_text": " ".join(texts)}]

    segments = []
    current = {"start_idx": None, "end_idx": None, "texts": []}
    prev_emb = None

    for i, t in enumerate(conversation):
        if t["speaker"] != "USER":
            continue
        txt = t["text"]
        emb = embed(txt)
        if current["start_idx"] is None:
            current = {"start_idx": i, "end_idx": i, "texts": [txt]}
            prev_emb = emb
        else:
            sim = cos_sim(prev_emb, emb)
            if sim is not None and sim < SIM_TOPIC_SHIFT:
                # close previous segment
                seg_txt = " ".join(current["texts"])
                segments.append({
                    "start_idx": current["start_idx"],
                    "end_idx": current["end_idx"],
                    "topics": extract_topics(seg_txt),
                    "repr_text": seg_txt
                })
                # start new
                current = {"start_idx": i, "end_idx": i, "texts": [txt]}
            else:
                current["end_idx"] = i
                current["texts"].append(txt)
            prev_emb = emb

    if current["start_idx"] is not None:
        seg_txt = " ".join(current["texts"])
        segments.append({
            "start_idx": current["start_idx"],
            "end_idx": current["end_idx"],
            "topics": extract_topics(seg_txt),
            "repr_text": seg_txt
        })

    return segments


# ---------------- Metrics ----------------

def cross_coherence(conversation: List[Dict[str, Any]]) -> Optional[float]:
    """Average similarity between USER turn and immediately following SYSTEM turn."""
    if embed("x") is None or util is None:
        return None
    sims = []
    for i in range(0, len(conversation) - 1, 2):
        a, b = conversation[i], conversation[i + 1]
        if a["speaker"] == "USER" and b["speaker"] == "SYSTEM":
            sims.append(cos_sim(embed(a["text"]), embed(b["text"])))
    return avg(sims)


def context_retention(conversation: List[Dict[str, Any]]) -> Optional[float]:
    """Average similarity between consecutive SYSTEM turns."""
    if embed("x") is None or util is None:
        return None
    sims = []
    prev_sys = None
    for t in conversation:
        if t["speaker"] != "SYSTEM":
            continue
        if prev_sys is not None:
            sims.append(cos_sim(embed(prev_sys["text"]), embed(t["text"])))
        prev_sys = t
    return avg(sims)


def jaccard_overlap(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_first_system_after(conversation, idx_user_end) -> Optional[int]:
    """Find index of the next SYSTEM message after a given user index."""
    for j in range(idx_user_end + 1, len(conversation)):
        if conversation[j]["speaker"] == "SYSTEM":
            return j
    return None


def topics_of_turn(text: str) -> List[str]:
    return extract_topics(text)


def system_alignment_with_topics(system_text: str, user_topics: List[str]) -> Optional[float]:
    """Similarity between system text and user topic bag."""
    if embed("x") is None or util is None:
        # fallback: Jaccard on token overlap of topics and system text tokens
        sys_topics = extract_topics(system_text)
        return jaccard_overlap(sys_topics, user_topics)
    return cos_sim(embed(system_text), embed(" ".join(user_topics)))


def compute_recovery_and_interference(conversation: List[Dict[str, Any]], user_segments: List[Dict[str, Any]]):
    """
    For each topic shift, compute:
    - recovered (bool)
    - recovery_delay (in system turns)
    - interference (old topic leakage before recovery)
    Returns summary + per-shift details.
    """
    details = []
    total_shifts = max(0, len(user_segments) - 1)
    recovered_count = 0
    delays = []
    interferences = []

    for k in range(total_shifts):
        prev_seg = user_segments[k]
        new_seg = user_segments[k + 1]

        # Determine reference user topics for new segment
        new_topics = new_seg["topics"]
        old_topics = prev_seg["topics"]

        # Find first system turn after the *last* USER of prev segment
        start_check_idx = find_first_system_after(conversation, new_seg["start_idx"])

        recovered = False
        delay_sys_turns = 0
        leakage_hits = 0
        sys_seen = 0

        if start_check_idx is not None:
            # Walk SYSTEM turns until clear recovery or end
            for j in range(start_check_idx, len(conversation)):
                t = conversation[j]
                if t["speaker"] != "SYSTEM":
                    continue
                sys_seen += 1

                # Alignment with new topics?
                align = system_alignment_with_topics(t["text"], new_topics) or 0.0
                # Interference with old topics?
                leak = system_alignment_with_topics(t["text"], old_topics) or 0.0

                if leak >= ALIGNMENT_THRESHOLD:
                    leakage_hits += 1

                if align >= ALIGNMENT_THRESHOLD:
                    recovered = True
                    delay_sys_turns = sys_seen  # number of SYSTEM turns till recovery
                    break

        if recovered:
            recovered_count += 1
            delays.append(delay_sys_turns)
        else:
            delays.append(None)

        # Interference as proportion of SYSTEM turns processed before recovery (or in window of 4)
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

    # Aggregates
    trr = recovered_count / total_shifts if total_shifts > 0 else None
    avg_delay = avg([d for d in delays if d is not None])
    avg_interference = avg(interferences) if interferences else None

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
    """
    CAS = weighted sum of normalized metrics:
      + topic_recovery_rate (higher better)
      + (1 - normalized avg_recovery_delay) (lower better)
      + (1 - normalized topic_interference) (lower better)
      + cross_coherence (higher better)
      + context_retention (higher better)
    """
    # Choose sane normalization ranges
    trr = metrics.get("topic_recovery_rate")              # already 0..1
    rd = normalize(metrics.get("avg_recovery_delay"), 1, 6, invert=True)  # 1..6 sys turns → shorter is better
    ti = normalize(metrics.get("topic_interference"), 0, 1, invert=True)  # 0..1 → lower is better
    cc = metrics.get("cross_coherence")                   # 0..1
    cr = metrics.get("context_retention")                 # 0..1

    parts = {
        "topic_recovery_rate": trr,
        "avg_recovery_delay": rd,
        "topic_interference": ti,
        "cross_coherence": cc,
        "context_retention": cr
    }

    # Normalize weights
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


# ---------------- Public API ----------------

def evaluate(conversation: List[Dict[str, Any]], true_genre: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entrypoint: compute all quantitative metrics + CAS.
    Returns:
      {
        ... metrics ...,
        "detail": {
            "user_segments": [...],
            "per_shift": [...]
        }
      }
    """
    # 1) Segment USER into topic segments
    user_segments = detect_user_segments(conversation)

    # 2) Cross-coherence and context retention
    cc = cross_coherence(conversation)
    cr = context_retention(conversation)

    # 3) Recovery & Interference over shifts
    ri = compute_recovery_and_interference(conversation, user_segments)

    metrics = {
        "topic_recovery_rate": ri["topic_recovery_rate"],
        "avg_recovery_delay": ri["avg_recovery_delay"],
        "topic_interference": ri["topic_interference"],
        "cross_coherence": cc,
        "context_retention": cr,
    }

    # 4) CAS
    cas = compute_cas(metrics)
    metrics["context_adaptation_score"] = cas

    # Legacy fields (kept for compatibility with your earlier scripts)
    metrics.update({
        "true_genre": true_genre,
        "num_topic_shifts": max(0, len(user_segments) - 1),
    })

    return {
        **metrics,
        "detail": {
            "user_segments": user_segments,
            "per_shift": ri["per_shift"]
        }
    }


def llm_judge(conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
    """LLM-as-judge for subjective evaluation."""
    convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in conversation])
    prompt = f"""
You are a dialogue evaluator. Here is a conversation between a USER and a SYSTEM:

{convo_text}

Evaluate the SYSTEM on a scale of 0–5 for:
1. Clarity (how clear and natural are the responses)
2. Politeness (tone, empathy)
3. Recovery ability (adapts to topic changes or corrections)
4. Context memory (retains user preferences)
5. Engagement (interactive and interesting)

Return strict JSON:
{{"clarity": X, "politeness": Y, "recovery": Z, "context_memory": W, "engagement": V}}
"""
    if MODE == "mock":
        return {"clarity": 4, "politeness": 5, "recovery": 3, "context_memory": 4, "engagement": 4}

    out = call_ollama(JUDGE_MODEL, prompt)
    try:
        return json.loads(out.strip())
    except Exception:
        return {"clarity": None, "politeness": None, "recovery": None, "context_memory": None, "engagement": None}
