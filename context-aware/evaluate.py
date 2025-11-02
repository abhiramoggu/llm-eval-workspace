# evaluate.py
import subprocess
import json
from config import MODE, JUDGE_MODEL

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None
_embedding_load_failed = False


def ensure_embedding_model():
    """Lazily load the embedding model; return None if unavailable."""
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


def get_topic_embedding(text):
    """Return normalized embedding for topic/context comparison."""
    model = ensure_embedding_model()
    if model is None:
        return None
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)


def semantic_similarity(text1, text2):
    """Compute cosine similarity between two texts."""
    if util is None:
        return None
    emb1, emb2 = get_topic_embedding(text1), get_topic_embedding(text2)
    if emb1 is None or emb2 is None:
        return None
    return float(util.cos_sim(emb1, emb2))


def extract_user_segments(conversation):
    """
    Split conversation into user/system segments by topic changes.
    We detect a change if semantic similarity between two consecutive
    USER utterances drops significantly.
    """
    if util is None or ensure_embedding_model() is None:
        # Treat entire conversation as a single segment when embeddings are absent.
        texts = [t["text"] for t in conversation if t["speaker"] == "USER"]
        return [texts] if texts else []

    user_segments = []
    current_segment = []
    prev_user_text = None

    for turn in conversation:
        if turn["speaker"] == "USER":
            if prev_user_text is not None:
                sim = semantic_similarity(prev_user_text, turn["text"])
                if sim is not None and sim < 0.55:  # threshold for topic shift
                    user_segments.append(current_segment)
                    current_segment = []
            current_segment.append(turn["text"])
            prev_user_text = turn["text"]

    if current_segment:
        user_segments.append(current_segment)
    return user_segments


def evaluate(conversation, true_genre):
    """
    Hybrid Evaluation:
    - Text/genre-based recovery metrics
    - Semantic coherence & context retention via embeddings
    - Topic segmentation & shift detection
    """
    text_recovery_success = False
    state_recovery_success = False
    rec_consistency = False
    recovery_turns = None

    # ---- (1) Classical Genre Recovery Metrics ----
    for i, turn in enumerate(conversation):
        if turn["speaker"] == "SYSTEM":
            if true_genre in turn["text"].lower() and not text_recovery_success:
                text_recovery_success = True
                recovery_turns = i // 2

            constraints = turn.get("constraints", {})
            if constraints.get("genre") == true_genre:
                state_recovery_success = True

            if true_genre in turn["text"].lower():
                rec_consistency = True

    # ---- (2) Embedding-based Context Consistency ----
    user_texts = [t["text"] for t in conversation if t["speaker"] == "USER"]
    system_texts = [t["text"] for t in conversation if t["speaker"] == "SYSTEM"]

    embeddings_ready = util is not None and ensure_embedding_model() is not None

    if embeddings_ready and len(user_texts) > 1:
        similarities = [
            semantic_similarity(user_texts[i], user_texts[i + 1])
            for i in range(len(user_texts) - 1)
        ]
        valid = [s for s in similarities if s is not None]
        user_coherence = sum(valid) / len(valid) if valid else None
    else:
        user_coherence = None

    if embeddings_ready and len(system_texts) > 1:
        similarities = [
            semantic_similarity(system_texts[i], system_texts[i + 1])
            for i in range(len(system_texts) - 1)
        ]
        valid = [s for s in similarities if s is not None]
        system_coherence = sum(valid) / len(valid) if valid else None
    else:
        system_coherence = None

    # ---- (3) Topic Shift & Recovery Coherence ----
    segments = extract_user_segments(conversation)
    recovery_alignment_scores = []

    for seg in segments:
        if not seg:
            continue
        # Find CRS responses after this segment
        idx = next(
            (i for i, t in enumerate(conversation)
             if t["speaker"] == "USER" and seg[-1] in t["text"]),
            None
        )
        if idx is not None:
            following_responses = [
                t["text"] for t in conversation[idx + 1:] if t["speaker"] == "SYSTEM"
            ]
            if following_responses:
                sim = semantic_similarity(seg[-1], following_responses[0])
                if sim is not None:
                    recovery_alignment_scores.append(sim)

    avg_recovery_alignment = (
        sum(recovery_alignment_scores) / len(recovery_alignment_scores)
        if recovery_alignment_scores else None
    )

    return {
        "true_genre": true_genre,
        "text_recovery_success": text_recovery_success,
        "state_recovery_success": state_recovery_success,
        "rec_consistency": rec_consistency,
        "recovery_turns": recovery_turns,
        "user_coherence": user_coherence,
        "system_coherence": system_coherence,
        "avg_recovery_alignment": avg_recovery_alignment,
        "num_topic_shifts": max(0, len(segments) - 1),
    }


def llm_judge(conversation):
    """
    LLM-as-judge evaluation for subjective dialogue quality.
    """
    convo_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in conversation])

    prompt = f"""
You are a dialogue evaluator. Here is a conversation between a USER and a SYSTEM:

{convo_text}

Evaluate the SYSTEM on a scale of 0–5 for:
1. Clarity (how clear and natural are the responses)
2. Politeness (tone, coherence, empathy)
3. Recovery ability (how well the system adapts when the user changes topics or corrects preferences)
4. Context memory (how well it retains information about user preferences)
5. Engagement (does it stay interactive and interesting)

Return strict JSON like:
{{"clarity": X, "politeness": Y, "recovery": Z, "context_memory": W, "engagement": V}}
"""

    if MODE == "mock":
        return {"clarity": 4, "politeness": 5, "recovery": 3, "context_memory": 4, "engagement": 4}

    else:
        process = subprocess.Popen(
            ["ollama", "run", JUDGE_MODEL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=prompt)
        if error:
            print("Error:", error)

        try:
            return json.loads(output.strip())
        except Exception as e:
            print("JSON parse error:", e)
            return {
                "clarity": None,
                "politeness": None,
                "recovery": None,
                "context_memory": None,
                "engagement": None
            }
