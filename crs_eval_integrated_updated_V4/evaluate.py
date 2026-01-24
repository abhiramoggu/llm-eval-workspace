# NOTE (LEGACY NAMING): Some function/field names use 'topic' from earlier iterations.
# In the thesis and docs, 'topic' refers to *catalog-grounded concepts* (field=value constraints),
# and 'topic shifts' correspond to changes in dominant constraint sets / preference focus.

"""
Hybrid evaluation for CRS:
- Topic extraction (LLM or heuristic)
- Topic extraction (LDA)
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
import os
import subprocess
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import numpy as np

from feature_extraction import DEFAULT_FEATURIZER, jaccard_similarity, cosine_similarity, copying_ratio

from config import (
    MODE,
    JUDGE_MODEL,
    TOPIC_EXTRACTOR_MODE,
    TOPIC_EXTRACTOR_MODEL,
    SIM_TOPIC_SHIFT,
    ALIGNMENT_THRESHOLD,
    TOPIC_JACCARD_SHIFT,
    CAS_WEIGHTS,
    TAS_WEIGHTS,
    RECOVERY_WINDOW,
    EMBEDDING_MODEL_NAME,
    VERBOSE_EVAL,
    LDA_MODEL_PATH,
    LDA_DICT_PATH,
    LDA_MODEL_DIR,
    USE_TRANSFORMER_EMBEDDER,
    EVAL_MODE,
    CONCEPT_FIELDS,
)
from dataset import MOVIE_DB # Import movie database for LDA training

TOPIC_CACHE: Dict[str, List[str]] = {}
SIM_CACHE: Dict[Tuple[str, str], float] = {}
_EMBEDDING_MODEL: Any = None
_EMBEDDING_UNAVAILABLE = object()
_EMBEDDING_BACKEND = "transformer"

# --- LDA Model Globals ---
_LDA_MODEL = None
_LDA_DICTIONARY = None
_LDA_LEMMATIZER = None
_STOP_WORDS = None
_GENSIM_AVAILABLE = False

try:
    from gensim import corpora
    from gensim.models import LdaModel
    from nltk.tokenize import word_tokenize
    _GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: `gensim` or `nltk` not installed. LDA topic extraction will not be available.")


class TfidfSentenceEmbedder:
    """Lightweight fallback embedder using sklearn's TF-IDF vectors."""

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = []
        for movie in MOVIE_DB.values():
            name = movie.get("name")
            plot = movie.get("plot")
            if name:
                corpus.append(name)
            if plot:
                corpus.append(plot)
        if not corpus:
            corpus = ["movie recommendation system"]

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english",
        )
        self.vectorizer.fit(corpus)

    def encode(self, texts: List[str], convert_to_tensor: bool = False):
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray()


def _build_tfidf_embedder() -> Optional[TfidfSentenceEmbedder]:
    try:
        embedder = TfidfSentenceEmbedder()
        print("Info: Using TF-IDF based sentence embedder as a fallback.")
        return embedder
    except Exception as exc:
        print(f"Warning: TF-IDF fallback embedder failed to initialize ({exc}).")
        return None


def ensure_embedding_model():
    """Lazy-loads the embedding backend, preferring sentence-transformers."""
    global _EMBEDDING_MODEL, _EMBEDDING_BACKEND
    if _EMBEDDING_MODEL is not None:
        return

    if USE_TRANSFORMER_EMBEDDER:
        torch_available = False
        torch_error = None
        try:
            import torch  # noqa: F401
            torch_available = True
        except Exception as exc:  # torch will raise if numpy ABI mismatches
            torch_error = exc

        if torch_available:
            try:
                from sentence_transformers import SentenceTransformer

                _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
                _EMBEDDING_BACKEND = "transformer"
                return
            except ImportError:
                print(
                    "Warning: sentence-transformers is not installed. "
                    "Falling back to TF-IDF embeddings."
                )
            except Exception as exc:
                print(
                    "Warning: sentence-transformers could not be initialized "
                    f"({exc}). Falling back to TF-IDF embeddings."
                )
        else:
            print(
                "Info: Torch backend unavailable for transformer embeddings "
                f"({torch_error}). Falling back to TF-IDF embeddings."
            )

    fallback = _build_tfidf_embedder()
    if fallback:
        _EMBEDDING_MODEL = fallback
        _EMBEDDING_BACKEND = "tfidf"
    else:
        _EMBEDDING_MODEL = _EMBEDDING_UNAVAILABLE
        _EMBEDDING_BACKEND = "unavailable"


def call_ollama_json(prompt: str, model_name: str) -> Optional[dict]:
    process = subprocess.Popen(
        ["ollama", "run", model_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output, error = process.communicate(input=prompt)
    err_text = (error or "").strip()
    if err_text and any(kw in err_text.lower() for kw in ["error", "failed"]):
        print(f"Ollama Error ({model_name}): {err_text}")
    if not output:
        return None
    try:
        return json.loads(output.strip())
    except Exception:
        print(f"Error parsing JSON from Ollama: {output}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        return None


def _preprocess_lda_text(text: str) -> List[str]:
    """Tokenize, lemmatize, and remove stopwords for LDA."""
    global _LDA_LEMMATIZER, _STOP_WORDS
    if _LDA_LEMMATIZER is None:
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords as nltk_stopwords
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("NLTK data not found. Downloading...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        _LDA_LEMMATIZER = WordNetLemmatizer()
        _STOP_WORDS = set(nltk_stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    lemmatized = [
        _LDA_LEMMATIZER.lemmatize(t) for t in tokens if t.isalpha() and t not in _STOP_WORDS and len(t) > 2
    ]
    return lemmatized

def ensure_lda_model():
    """Loads the pre-trained LDA model and dictionary from disk."""
    global _LDA_MODEL, _LDA_DICTIONARY
    if not _GENSIM_AVAILABLE or _LDA_MODEL is not None:
        return

    if os.path.exists(LDA_MODEL_PATH) and os.path.exists(LDA_DICT_PATH):
        if VERBOSE_EVAL:
            print("--- Loading pre-trained LDA model ---")
        _LDA_MODEL = LdaModel.load(LDA_MODEL_PATH)
        _LDA_DICTIONARY = corpora.Dictionary.load(LDA_DICT_PATH)
    else:
        print("\n" + "="*60)
        print("ERROR: Pre-trained LDA model not found.")
        print(f"Please run `python train_lda.py` once to generate the model files.")
        print("="*60 + "\n")

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
        err_text = (err or "").strip()
        if err_text and any(kw in err_text.lower() for kw in ["error", "failed"]):
            print(f"Ollama Error ({model_name}): {err_text}")
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Is Ollama installed and in your PATH?")
        return ""
    return (out or "").strip()


def extract_topics_llm(text: str, model: str = TOPIC_EXTRACTOR_MODEL, top_k: int = 5) -> List[str]:
    """LLM-based extractor with topic filtering."""
    if text in TOPIC_CACHE:
        return TOPIC_CACHE[text][:top_k]

    prompt = f"""
From the user's message below, extract up to 5 key thematic topics. These can be genres, emotions, concepts, or tones.
Your response MUST be a single, valid JSON array of strings, and nothing else. Do not add any commentary.
For example: ["romance", "space opera", "nostalgia", "fear"]
---
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


def extract_topics_lda(text: str, top_k: int = 5) -> List[str]:
    """Infers topics for a given text using the pre-trained LDA model."""
    ensure_lda_model()
    if not _LDA_MODEL or not _LDA_DICTIONARY:
        if _GENSIM_AVAILABLE:
            print("Warning: LDA model not available. Falling back to heuristic topic extraction.")
        return extract_topics_heuristic(text, top_k)

    if text in TOPIC_CACHE:
        return TOPIC_CACHE[text][:top_k]

    processed_text = _preprocess_lda_text(text)
    bow = _LDA_DICTIONARY.doc2bow(processed_text)
    
    topic_dist = _LDA_MODEL.get_document_topics(bow, minimum_probability=0.1)
    if not topic_dist:
        return []

    # Get the most significant topic and its top words
    best_topic_id = max(topic_dist, key=lambda item: item[1])[0]
    topics = [word for word, _ in _LDA_MODEL.show_topic(best_topic_id, topn=top_k)]
    
    TOPIC_CACHE[text] = topics
    return topics

def extract_topics(text: str) -> List[str]:
    if TOPIC_EXTRACTOR_MODE == "llm":
        return extract_topics_llm(text)
    if TOPIC_EXTRACTOR_MODE == "lda":
        return extract_topics_lda(text)
    return extract_topics_heuristic(text)


def topics_to_text(topics: List[str]) -> str:
    return ", ".join(topics) if topics else ""


def embedding_similarity(text_a: str, text_b: str) -> float:
    """Calculate semantic similarity using a sentence-transformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            ensure_embedding_model()
        except Exception as exc:
            print(
                "Warning: embedding backend failed to initialize; "
                f"falling back to lexical similarity ({exc})."
            )
            _EMBEDDING_MODEL = _EMBEDDING_UNAVAILABLE
    key = tuple(sorted((text_a, text_b)))
    if key in SIM_CACHE:
        return SIM_CACHE[key]

    # After attempting to load, check the status of _EMBEDDING_MODEL.
    # This correctly handles the case where the model was just loaded.
    if _EMBEDDING_MODEL is _EMBEDDING_UNAVAILABLE or not hasattr(_EMBEDDING_MODEL, "encode"):
        if _EMBEDDING_MODEL is not _EMBEDDING_UNAVAILABLE:
            print("Warning: sentence-transformers is not installed or model failed to load. Similarity metrics will be degraded.")
        sim = SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()
        SIM_CACHE[key] = sim
        return sim

    embeddings = _EMBEDDING_MODEL.encode([text_a, text_b], convert_to_tensor=False)
    vec_a = _to_numpy_vector(embeddings[0])
    vec_b = _to_numpy_vector(embeddings[1])
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) or 1e-8
    sim = float(np.dot(vec_a, vec_b) / denom)
    SIM_CACHE[key] = sim
    return sim


def _to_numpy_vector(vec: Any) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        return vec
    if hasattr(vec, "detach"):
        return vec.detach().cpu().numpy()
    if hasattr(vec, "cpu"):
        return np.asarray(vec.cpu())
    if hasattr(vec, "numpy"):
        return np.asarray(vec.numpy())
    if isinstance(vec, (list, tuple)):
        return np.asarray(vec, dtype=float)
    return np.asarray(vec, dtype=float)


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
            curr_topics = extract_topics(txt)
            sim = embedding_similarity(base, txt)
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


def _get_user_meta(turn: Dict[str, Any]) -> Dict[str, Any]:
    meta = turn.get("user_meta")
    if isinstance(meta, dict):
        return meta
    meta = turn.get("meta")
    if isinstance(meta, dict):
        return meta
    return {}


def _has_user_meta(turn: Dict[str, Any]) -> bool:
    return isinstance(turn.get("user_meta"), dict) or isinstance(turn.get("meta"), dict)


def _shift_points_with_presence(
    conversation: List[Dict[str, Any]],
    shift_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[int], bool]:
    """Return shift points and whether simulator metadata was present."""
    shift_points: List[int] = []
    meta_present = False

    if shift_events is not None:
        meta_present = True
        for ev in shift_events:
            if not isinstance(ev, dict):
                continue
            turn = ev.get("turn")
            if isinstance(turn, int) and turn >= 1:
                shift_points.append(turn)

    user_turn = 0
    for t in conversation:
        if t.get("speaker") != "USER":
            continue
        user_turn += 1
        if _has_user_meta(t):
            meta_present = True
        meta = _get_user_meta(t)
        if meta.get("shift_event") or meta.get("is_shift"):
            shift_points.append(user_turn)

    return sorted(set(shift_points)), meta_present


def extract_shift_points_from_conversation(
    conversation: List[Dict[str, Any]],
    shift_events: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Return USER-turn indices (1-based) where the simulator marked a shift.

    This is preferred over heuristic segmentation when available because it provides
    ground-truth shift locations generated by the simulator policy.
    """
    shift_points, _ = _shift_points_with_presence(conversation, shift_events=shift_events)
    return shift_points


def build_user_segments_from_shift_points(conversation: List[Dict[str, Any]], shift_points: List[int]) -> List[Dict[str, Any]]:
    """Build USER segments using simulator-provided shift points.

    Segments are over USER turns (not raw message indices). Each segment dict matches
    the structure expected by compute_recovery_and_interference().
    """
    # Map USER turn index -> conversation list index
    user_to_conv_idx = {}
    user_turn = 0
    for i, t in enumerate(conversation):
        if t.get("speaker") == "USER":
            user_turn += 1
            user_to_conv_idx[user_turn] = i

    # Segment boundaries in USER-turn space
    boundaries = [1] + [sp for sp in shift_points if sp >= 1] + [user_turn + 1]
    segments = []
    for seg_id in range(len(boundaries) - 1):
        u_start = boundaries[seg_id]
        u_end_excl = boundaries[seg_id + 1]  # exclusive
        if u_start > user_turn:
            continue
        u_end_excl = min(u_end_excl, user_turn + 1)

        conv_start_idx = user_to_conv_idx[u_start]
        conv_end_idx = user_to_conv_idx[u_end_excl - 1]  # last USER turn conv index
        # Collect USER utterances in this segment and extract concept set ("topics" in code)
        seg_texts = []
        seg_meta = []
        for u in range(u_start, u_end_excl):
            ci = user_to_conv_idx[u]
            seg_texts.append(conversation[ci]["text"])
            seg_meta.append(_get_user_meta(conversation[ci]))
        joined = " ".join(seg_texts)
        seg_topics = _filter_topics(extract_topics(joined))

        # Choose representative focus field if present
        focus_fields = [m.get("focus_field") for m in seg_meta if m.get("focus_field")]
        focus_field = focus_fields[-1] if focus_fields else None

        segments.append({
            "seg_id": seg_id,
            "start_user_turn": u_start,
            "end_user_turn_excl": u_end_excl,
            "start_idx": conv_start_idx,
            "end_idx": conv_end_idx,
            "topics": seg_topics,
            "focus_field": focus_field,
        })
    return segments
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


def compute_cas(metrics: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
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
            weighted_val = w * v
            score += weighted_val
            denom += w
            parts[f"cas_{k}_weighted"] = weighted_val # Store the weighted component

    if denom == 0:
        return {"context_adaptation_score": None, "cas_components": parts}

    return {"context_adaptation_score": score / denom, "cas_components": parts}


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


def _get_user_preferences_summary(conversation: List[Dict[str, Any]]) -> str:
    """Extracts a summary of user preferences from their utterances."""
    prefs = []
    for turn in conversation:
        if turn["speaker"] == "USER":
            # Simple heuristic: look for phrases that state a preference
            if re.search(r"i want|i'm in the mood for|how about|looking for", turn["text"], re.IGNORECASE):
                prefs.append(turn["text"])
    return "\n".join(prefs) if prefs else "No specific preferences stated."


def llm_judge(conversation: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """LLM-as-judge for qualitative metrics."""
    convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in conversation])
    user_prefs = _get_user_preferences_summary(conversation)

    prompt = f"""
[Task Description]
You will be provided with a conversation between a User (annotated as "User") and a Conversational Recommender System (annotated as "Recommender") discussing movie recommendations.
Your task is to evaluate and rate the performance of the Conversational Recommender System based on three specific metrics.

[Evaluation Criteria]
######
Metric 1 - Proactiveness (1-5): This refers to the system’s capability to take initiative in guiding the conversation, asking relevant questions, and making suggestions to actively uncover and clarify the user’s preferences
1: The Recommender is not proactive.
2: The Recommender is slightly proactive.
3: The Recommender is moderately proactive.
4: The Recommender is mostly proactive.
5: The Recommender is completely proactive.

######
Metric 2 - Coherence (1-5): This refers to the system's proficiency in maintaining logical consistency throughout the entire conversation, avoiding contradictions with previous statements, building upon earlier discussed preferences without unnecessary repetition, and providing contextually appropriate responses without abrupt transitions or disjointed exchanges.
1: The Recommender's responses are incoherent.
2: The Recommender's responses are slightly coherent.
3: The Recommender's responses are moderately coherent.
4: The Recommender's responses are mostly coherent.
5: The Recommender's responses are completely coherent.

######
Metric 3 - Personalization (1-5): This refers to the system’s capability to engage in fluid interactions with users, providing linguistically natural responses that are contextually related to previous interactions, without abrupt transitions or disjointed exchanges.
1: The Recommender does not fulfill the user's preferences.
2: The Recommender slightly fulfills the user's preferences.
3: The Recommender moderately fulfills the user's preferences.
4: The Recommender mostly fulfills the user's preferences.
5: The Recommender consistently fulfills the user's preferences.

[Inputs]:
User preferences:
{user_prefs}

Conversation History:
{convo_text}

[Output Format]
You MUST provide your response *only* in the following format. Replace "[Score]" with the appropriate number. Do not include any other text, explanations, introductory phrases, or closing remarks.

- Proactiveness: [Score]
- Coherence: [Score]
- Personalization: [Score]
"""

    if MODE == "mock":
        return {"proactiveness": 4, "coherence": 4, "personalization": 3}

    # Use the standard call_ollama since the output is not JSON but a simple text format.
    output = call_ollama(JUDGE_MODEL, prompt)
    scores = {}
    if output:
        try:
            for line in output.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace('-', '').replace(' ', '')
                    # Extract digits from the value string
                    score_val = re.search(r'\d+', value)
                    if score_val:
                        scores[key] = int(score_val.group())
        except Exception as e:
            print(f"LLM Judge Parsing Error: {e}\nRaw Output:\n{output}")

    expected_keys = ["proactiveness", "coherence", "personalization"]
    return {key: scores.get(key) for key in expected_keys}



# ---------------- Public API ----------------

def evaluate_legacy(
    conversation: List[Dict[str, Any]],
    true_genre: Optional[str] = None,
    shift_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if VERBOSE_EVAL:
        print(f"\n[EVAL] Starting evaluation: {len(conversation)} turns, true_genre={true_genre}")

    shift_points, meta_present = _shift_points_with_presence(conversation, shift_events=shift_events)
    if meta_present:
        user_segments = build_user_segments_from_shift_points(conversation, shift_points)
    else:
        user_segments = detect_user_segments(conversation)
    cc = cross_coherence(conversation)
    cr = context_retention(conversation)
    ri = compute_recovery_and_interference(conversation, user_segments)
    turn_alignments = compute_turn_alignments(conversation, user_segments)

    # Count topics per system turn
    system_topic_counts = []
    for turn in conversation:
        if turn["speaker"] == "SYSTEM":
            topics = extract_topics(turn["text"])
            system_topic_counts.append(len(topics))

    metrics = {
        "topic_recovery_rate": ri["topic_recovery_rate"],
        "avg_recovery_delay": ri["avg_recovery_delay"],
        "topic_interference": ri["topic_interference"],
        "cross_coherence": cc,
        "context_retention": cr,
    }

    cas_results = compute_cas(metrics)
    metrics["context_adaptation_score"] = cas_results["context_adaptation_score"]

    metrics.update({
        "true_genre": true_genre,
        "num_topic_shifts": max(0, len(user_segments) - 1),
    })

    result = {
        **metrics,
        "detail": {
            "user_segments": user_segments,
            "per_shift": ri["per_shift"],
            "turn_alignments": turn_alignments, # This is a list of dicts
            "system_topic_counts": system_topic_counts, # This is a list of ints
            "cas_components": cas_results["cas_components"]
        }
    }
    return result


# ---------------------------------------------------------------------------
# Optional Preference Coverage Evaluation
# ---------------------------------------------------------------------------

class PreferenceCoverageMetric:
    """
    Utility helper to compute preference coverage, recall, precision, and NDCG
    using recommendation logs stored per conversation turn.

    This adapts a public GitHub snippet to work within this project without
    touching the primary evaluation pipeline. Usage:

    >>> pcm = PreferenceCoverageMetric(turn_num=20, k_values=[1, 5, 10])
    >>> pcm.update_dialogue(recommendations_per_turn, ground_truth_titles)
    >>> summary = pcm.preference_coverage_report()
    """

    def __init__(self, turn_num: int, k_values: Optional[List[int]] = None):
        self.turn_num = turn_num
        self.k_values = k_values or [1, 5, 10, 20, 50]
        self.reset()

    def reset(self):
        self.dialogue_count = 0
        self.dialogue_count_for_rp = 1  # avoid divide-by-zero
        self.turn_metrics: Dict[int, Dict[str, float]] = {}
        self.ndcg_all = 0.0
        for turn in range(1, self.turn_num + 1):
            self.turn_metrics[turn] = {"preference_coverage": 0.0}
            for k in self.k_values:
                self.turn_metrics[turn][f"recall@{k}"] = 0.0
                self.turn_metrics[turn][f"precision@{k}"] = 0.0

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    def compute_recall(self, recommended: List[str], gt_labels: List[str], k: int) -> float:
        recommended_at_k = recommended[:k]
        correct = set(recommended_at_k) & set(gt_labels)
        return self._safe_div(len(correct), len(gt_labels))

    def compute_precision(self, recommended: List[str], gt_labels: List[str], k: int) -> float:
        recommended_at_k = recommended[:k]
        relevant = set(recommended_at_k) & set(gt_labels)
        return self._safe_div(len(relevant), k)

    def compute_best_rank(self, recommended: List[str], gt_labels: List[str], best_ranks: Dict[str, int]) -> Dict[str, int]:
        for label in gt_labels:
            if label in recommended:
                current_rank = recommended.index(label) + 1
                if best_ranks[label] == -1 or current_rank < best_ranks[label]:
                    best_ranks[label] = current_rank
        return best_ranks

    def compute_ndcg(self, best_ranks: Dict[str, int]) -> float:
        import math

        total_ndcg = 0.0
        num_labels = len(best_ranks)
        for rank in best_ranks.values():
            if rank != -1:
                dcg = 1 / math.log2(rank + 1)
            else:
                dcg = 0.0
            idcg = 1 / math.log2(1 + 1)
            total_ndcg += self._safe_div(dcg, idcg)
        ndcg = self._safe_div(total_ndcg, num_labels)
        self.ndcg_all += ndcg
        return ndcg

    def update_dialogue(self, recommendations_per_turn: List[List[str]], gt_labels: List[str]) -> None:
        """
        recommendations_per_turn: ordered list of per-turn recommendation lists.
        gt_labels: list of gold movie IDs/titles.
        """
        if not gt_labels:
            return

        cumulative: List[str] = []
        best_ranks = {label: -1 for label in gt_labels}

        for idx in range(1, self.turn_num + 1):
            recommended = recommendations_per_turn[idx - 1] if idx - 1 < len(recommendations_per_turn) else []
            cumulative.extend(recommended)
            cumulative = list(dict.fromkeys(cumulative))  # deduplicate while keeping order

            coverage = self._safe_div(len(set(cumulative) & set(gt_labels)), len(gt_labels))
            self.turn_metrics[idx]["preference_coverage"] += coverage

            for k in self.k_values:
                recall_k = self.compute_recall(recommended, gt_labels, k)
                precision_k = self.compute_precision(recommended, gt_labels, k)
                self.turn_metrics[idx][f"recall@{k}"] += recall_k
                self.turn_metrics[idx][f"precision@{k}"] += precision_k

            best_ranks = self.compute_best_rank(recommended, gt_labels, best_ranks)

        self.compute_ndcg(best_ranks)
        self.dialogue_count += 1
        self.dialogue_count_for_rp += 1

    def preference_coverage_report(self) -> List[Dict[str, float]]:
        report = [{"total_dialogues": self.dialogue_count, "average_ndcg": self._safe_div(self.ndcg_all, max(1, self.dialogue_count))}]
        for idx in range(1, self.turn_num + 1):
            report.append({
                "turn": idx,
                "preference_coverage": self._safe_div(self.turn_metrics[idx]["preference_coverage"], max(1, self.dialogue_count)),
            })
        return report

    def recall_precision_report(self) -> List[Dict[str, Any]]:
        report: List[Dict[str, Any]] = [{"total_dialogues": self.dialogue_count_for_rp - 1}]
        for idx in range(1, self.turn_num + 1):
            metrics = {}
            for k in self.k_values:
                metrics[f"recall@{k}"] = self._safe_div(self.turn_metrics[idx][f"recall@{k}"], self.dialogue_count_for_rp)
                metrics[f"precision@{k}"] = self._safe_div(self.turn_metrics[idx][f"precision@{k}"], self.dialogue_count_for_rp)
            report.append({"turn": idx, "metrics": metrics})
        return report


def evaluate_preference_coverage_logs(
    base_dir: str,
    datasets: List[str],
    models: List[str],
    turn_num: int = 20,
    k_values: Optional[List[int]] = None,
    target_key: str = "target",
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Convenience wrapper that scans a directory structure containing saved
    recommendation logs in the shape:

        <base_dir>/<model>/<dataset>/eval/*.json

    Each eval JSON is expected to follow the schema from the original snippet:
    the first element is a metadata dict that includes the ground-truth labels
    (stored under `target_key`, `target_movies_sampled`, or
    `target_movies_not_sampled`), and subsequent elements contain recommendation
    lists with keys `recommended` and `turn num`.

    Returns a nested dict: reports[model][dataset] -> preference coverage report.
    """
    reports: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    for model in models:
        model_reports: Dict[str, List[Dict[str, float]]] = {}
        for dataset in datasets:
            metric = PreferenceCoverageMetric(turn_num, k_values)
            eval_path = os.path.join(base_dir, model, dataset, "eval")
            if not os.path.isdir(eval_path):
                continue

            filenames = sorted(os.listdir(eval_path))
            iterator = filenames
            try:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(filenames, desc=f"{model}-{dataset}")
            except Exception:
                pass

            for filename in iterator:
                full_path = os.path.join(eval_path, filename)
                try:
                    with open(full_path, "r", encoding="utf-8") as fp:
                        content = json.load(fp)
                except Exception:
                    continue

                if not content:
                    continue

                header = content[0]
                gt_labels = header.get(target_key) or header.get("target_movies_sampled") or header.get("target_movies_not_sampled") or header.get("ground_truth") or []
                if not isinstance(gt_labels, list) or not gt_labels:
                    continue

                per_turn_recs: List[List[str]] = []
                for turn_entry in content[1:]:
                    recommended = turn_entry.get("recommended", [])
                    if isinstance(recommended, list):
                        per_turn_recs.append(recommended)
                metric.update_dialogue(per_turn_recs, gt_labels)

            model_reports[dataset] = metric.preference_coverage_report()
        if model_reports:
            reports[model] = model_reports
    return reports


# ============================================================
# Grounded TAS evaluation (preferred; used when EVAL_MODE="grounded")
# ============================================================

def _pair_turns(conversation):
    # Return list of (user_entry, system_entry) for turn-level scoring.
    pairs = []
    for i in range(0, len(conversation) - 1, 2):
        u = conversation[i]
        s = conversation[i + 1]
        if (u.get("speaker") == "USER") and (s.get("speaker") == "SYSTEM"):
            pairs.append((u, s))
    return pairs


def _shift_points_from_meta(
    conversation: List[Dict[str, Any]],
    shift_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[int], bool]:
    # Shift points as 1-based USER turns when simulator metadata is available.
    return _shift_points_with_presence(conversation, shift_events=shift_events)


def _user_turn_index_map(conversation: List[Dict[str, Any]]) -> Tuple[Dict[int, int], int]:
    mapping: Dict[int, int] = {}
    user_turn = 0
    for idx, turn in enumerate(conversation):
        if turn.get("speaker") == "USER":
            user_turn += 1
            mapping[idx] = user_turn
    return mapping, user_turn


def _segments_from_heuristics(conversation: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], List[int]]:
    # Convert heuristic USER segments to 1-based turn index ranges.
    segments = []
    shift_points = []
    user_segments = detect_user_segments(conversation)
    conv_to_user, _ = _user_turn_index_map(conversation)
    for seg in user_segments:
        st = conv_to_user.get(seg.get("start_idx"))
        en = conv_to_user.get(seg.get("end_idx"))
        if st is None or en is None:
            continue
        segments.append((st, en))
    for seg in segments[1:]:
        shift_points.append(seg[0])
    return segments, shift_points


def _segments_from_shift_points(num_turns, shift_points):
    # Segments as list of (start, end) inclusive indices (1-based).
    if num_turns <= 0:
        return []
    starts = [1] + [sp for sp in shift_points if 1 <= sp <= num_turns]
    starts = sorted(set(starts))
    segments = []
    for i, st in enumerate(starts):
        en = (starts[i + 1] - 1) if (i + 1) < len(starts) else num_turns
        if st <= en:
            segments.append((st, en))
    return segments


def _aggregate_concept_sets(texts):
    agg = set()
    for t in texts:
        agg |= DEFAULT_FEATURIZER.concept_set(t, fields=CONCEPT_FIELDS)
    return agg


def _aggregate_vectors(texts):
    vecs = [DEFAULT_FEATURIZER.attribute_vector(t) for t in texts]
    if not vecs:
        import numpy as np
        return np.zeros(DEFAULT_FEATURIZER.vocab_size, dtype=float)
    import numpy as np
    return np.sum(np.stack(vecs, axis=0), axis=0)


def evaluate_grounded(
    conversation: List[Dict[str, Any]],
    shift_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    # Compute grounded turn-level TAS + segment metrics + recovery metrics.
    pairs = _pair_turns(conversation)
    T = len(pairs)
    if T == 0:
        return {
            "cross_coherence": 0.0,
            "context_retention": 0.0,
            "topic_interference": 0.0,
            "context_adaptation_score": 0.0,
            "topic_recovery_rate": 0.0,
            "avg_recovery_delay": None,
            "missing_concepts_total": 0,
            "hallucinated_concepts_total": 0,
            "missing_concepts_per_turn": [],
            "hallucinated_concepts_per_turn": [],
            "missing_concepts_topk": [],
            "hallucinated_concepts_topk": [],
            "detail": {"turn_level": [], "segments": [], "shift_points": []},
        }

    alpha = TAS_WEIGHTS.get("alpha", 1.0)
    beta = TAS_WEIGHTS.get("beta", 1.0)
    gamma = TAS_WEIGHTS.get("gamma", 1.0)

    cc_list, cr_list, i_list, tas_list = [], [], [], []
    turn_detail = []
    missing_per_turn = []
    hallucinated_per_turn = []
    missing_counter: Counter = Counter()
    hallucinated_counter: Counter = Counter()

    for t_idx, (u, s) in enumerate(pairs, start=1):
        u_text = u.get("text", "")
        s_text = s.get("text", "")

        concepts_u = DEFAULT_FEATURIZER.concept_set(u_text, fields=CONCEPT_FIELDS)
        concepts_s = DEFAULT_FEATURIZER.concept_set(s_text, fields=CONCEPT_FIELDS)
        cc = jaccard_similarity(concepts_u, concepts_s)

        vec_u = DEFAULT_FEATURIZER.attribute_vector(u_text)
        vec_s = DEFAULT_FEATURIZER.attribute_vector(s_text)
        cr = cosine_similarity(vec_u, vec_s)

        it = copying_ratio(s_text, u_text)
        missing = sorted(list(concepts_u - concepts_s))
        hallucinated = sorted(list(concepts_s - concepts_u))

        tas = (alpha * cc) + (beta * cr) - (gamma * it)
        tas = float(max(-1.0, min(1.0, tas)))

        cc_list.append(cc); cr_list.append(cr); i_list.append(it); tas_list.append(tas)
        missing_per_turn.append(missing)
        hallucinated_per_turn.append(hallucinated)
        missing_counter.update(missing)
        hallucinated_counter.update(hallucinated)
        turn_detail.append({
            "turn": t_idx,
            "cc": cc,
            "cr": cr,
            "i": it,
            "tas": tas,
            "user_meta": u.get("user_meta", {}),
            "missing_concepts": missing,
            "hallucinated_concepts": hallucinated,
        })

    import numpy as np
    cc_mean = float(np.mean(cc_list))
    cr_mean = float(np.mean(cr_list))
    i_mean = float(np.mean(i_list))
    tas_mean = float(np.mean(tas_list))

    shift_points, meta_present = _shift_points_from_meta(conversation, shift_events=shift_events)
    if meta_present:
        segments = _segments_from_shift_points(T, shift_points)
    else:
        segments, heuristic_shift_points = _segments_from_heuristics(conversation)
        if not segments:
            segments = _segments_from_shift_points(T, [])
        if heuristic_shift_points:
            shift_points = heuristic_shift_points

    segment_detail = []
    for (st, en) in segments:
        user_texts = [pairs[i-1][0].get("text","") for i in range(st, en+1)]
        sys_texts  = [pairs[i-1][1].get("text","") for i in range(st, en+1)]
        cc_seg = jaccard_similarity(_aggregate_concept_sets(user_texts), _aggregate_concept_sets(sys_texts))
        seg_concepts_u = _aggregate_concept_sets(user_texts)
        seg_concepts_s = _aggregate_concept_sets(sys_texts)
        seg_missing = sorted(list(seg_concepts_u - seg_concepts_s))
        seg_hallucinated = sorted(list(seg_concepts_s - seg_concepts_u))
        cr_seg = cosine_similarity(_aggregate_vectors(user_texts), _aggregate_vectors(sys_texts))
        i_seg = float(np.mean([copying_ratio(pairs[i-1][1].get("text",""), pairs[i-1][0].get("text","")) for i in range(st, en+1)]))
        tas_seg = (alpha*cc_seg) + (beta*cr_seg) - (gamma*i_seg)
        tas_seg = float(max(-1.0, min(1.0, tas_seg)))
        segment_detail.append({
            "start_turn": st,
            "end_turn": en,
            "missing_concepts": seg_missing,
            "hallucinated_concepts": seg_hallucinated,
            "cc": cc_seg,
            "cr": cr_seg,
            "i": i_seg,
            "tas": tas_seg,
        })

    threshold = ALIGNMENT_THRESHOLD
    delays = []
    for sp in shift_points:
        found = None
        for d in range(0, max(1, RECOVERY_WINDOW)):
            idx = sp + d
            if idx <= T and cc_list[idx-1] >= threshold:
                found = d
                break
        if found is not None:
            delays.append(found)

    recovery_rate = float(len(delays) / len(shift_points)) if shift_points else 0.0
    avg_delay = float(np.mean(delays)) if delays else None
    missing_total = int(sum(len(items) for items in missing_per_turn))
    hallucinated_total = int(sum(len(items) for items in hallucinated_per_turn))
    missing_topk = [{"concept": c, "count": int(n)} for c, n in missing_counter.most_common(10)]
    hallucinated_topk = [{"concept": c, "count": int(n)} for c, n in hallucinated_counter.most_common(10)]

    return {
        "cross_coherence": cc_mean,
        "context_retention": cr_mean,
        "topic_interference": i_mean,
        "context_adaptation_score": tas_mean,
        # Thesis terminology aliases (keep legacy keys for backward-compatibility)
        "concept_overlap_mean": cc_mean,
        "constraint_similarity_mean": cr_mean,
        "copying_penalty_mean": i_mean,
        "adaptation_score_mean": tas_mean,
        "weighted_constraint_similarity_mean": cr_mean,  # alias: WCS mean
        "trajectory_adaptation_score_mean": tas_mean,    # alias: TAS mean
        "concept_recovery_rate": recovery_rate,
        "avg_concept_recovery_delay": avg_delay,
        "topic_recovery_rate": recovery_rate,
        "avg_recovery_delay": avg_delay,
        "missing_concepts_total": missing_total,
        "hallucinated_concepts_total": hallucinated_total,
        "missing_concepts_per_turn": missing_per_turn,
        "hallucinated_concepts_per_turn": hallucinated_per_turn,
        "missing_concepts_topk": missing_topk,
        "hallucinated_concepts_topk": hallucinated_topk,
        "detail": {
            "turn_level": turn_detail,
            "segments": segment_detail,
            "shift_points": shift_points,
            "num_turns": T,
        },
    }


def _compute_concept_deltas_from_pairs(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    missing_per_turn = []
    hallucinated_per_turn = []
    missing_counter: Counter = Counter()
    hallucinated_counter: Counter = Counter()
    for u, s in pairs:
        u_text = u.get("text", "")
        s_text = s.get("text", "")
        concepts_u = DEFAULT_FEATURIZER.concept_set(u_text, fields=CONCEPT_FIELDS)
        concepts_s = DEFAULT_FEATURIZER.concept_set(s_text, fields=CONCEPT_FIELDS)
        missing = sorted(list(concepts_u - concepts_s))
        hallucinated = sorted(list(concepts_s - concepts_u))
        missing_per_turn.append(missing)
        hallucinated_per_turn.append(hallucinated)
        missing_counter.update(missing)
        hallucinated_counter.update(hallucinated)
    missing_total = int(sum(len(items) for items in missing_per_turn))
    hallucinated_total = int(sum(len(items) for items in hallucinated_per_turn))
    missing_topk = [{"concept": c, "count": int(n)} for c, n in missing_counter.most_common(10)]
    hallucinated_topk = [{"concept": c, "count": int(n)} for c, n in hallucinated_counter.most_common(10)]
    return {
        "missing_concepts_total": missing_total,
        "hallucinated_concepts_total": hallucinated_total,
        "missing_concepts_per_turn": missing_per_turn,
        "hallucinated_concepts_per_turn": hallucinated_per_turn,
        "missing_concepts_topk": missing_topk,
        "hallucinated_concepts_topk": hallucinated_topk,
    }


def _add_metric_aliases(metrics: Dict[str, Any]) -> None:
    cc = metrics.get("concept_overlap_mean")
    if cc is None:
        cc = metrics.get("cross_coherence")
    cr = metrics.get("constraint_similarity_mean")
    if cr is None:
        cr = metrics.get("context_retention")
    cp = metrics.get("copying_penalty_mean")
    if cp is None:
        cp = metrics.get("topic_interference")
    tas = metrics.get("adaptation_score_mean")
    if tas is None:
        tas = metrics.get("context_adaptation_score")

    if cc is not None:
        metrics.setdefault("concept_overlap_mean", cc)
    if cr is not None:
        metrics.setdefault("constraint_similarity_mean", cr)
        metrics.setdefault("weighted_constraint_similarity_mean", cr)
    if cp is not None:
        metrics.setdefault("copying_penalty_mean", cp)
    if tas is not None:
        metrics.setdefault("adaptation_score_mean", tas)
        metrics.setdefault("trajectory_adaptation_score_mean", tas)

    if "concept_recovery_rate" not in metrics and "topic_recovery_rate" in metrics:
        metrics["concept_recovery_rate"] = metrics.get("topic_recovery_rate")
    if "avg_concept_recovery_delay" not in metrics and "avg_recovery_delay" in metrics:
        metrics["avg_concept_recovery_delay"] = metrics.get("avg_recovery_delay")


def _ensure_concept_delta_outputs(metrics: Dict[str, Any], conversation: List[Dict[str, Any]]) -> None:
    if "missing_concepts_total" in metrics and "hallucinated_concepts_total" in metrics:
        return
    pairs = _pair_turns(conversation)
    metrics.update(_compute_concept_deltas_from_pairs(pairs))


def evaluate(conversation, true_genre=None, mode=None):
    # Unified entry-point used by batch_run.py.
    shift_events = None
    if isinstance(conversation, dict):
        payload = conversation
        conversation = payload.get("conversation", [])
        shift_events = payload.get("shift_events")
        if true_genre is None:
            true_genre = payload.get("true_genre")
    mode = mode or EVAL_MODE
    if mode == "legacy":
        result = evaluate_legacy(conversation, true_genre, shift_events=shift_events)
    else:
        result = evaluate_grounded(conversation, shift_events=shift_events)
    _add_metric_aliases(result)
    _ensure_concept_delta_outputs(result, conversation)
    return result