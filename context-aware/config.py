# config.py
import os
from dataset import get_available_genres

# ============ MODES ============
# "mock" -> no LLM calls, simple placeholders
# "ollama" -> uses local Ollama models for USER / CRS / Judge / Topic extraction
MODE = os.getenv("SIM_MODE", "ollama")

# Fail safe: alert if MODE differs from supported values
if MODE not in {"ollama", "mock"}:
    raise ValueError(f"Unsupported SIM_MODE '{MODE}'. Use 'ollama' or 'mock'.")

# ============ DATA ============
KNOWN_GENRES = set(get_available_genres())

# Allow switching off transformer embeddings when dependencies aren't available
USE_TRANSFORMER_EMBEDDER = bool(int(os.getenv("USE_TRANSFORMER_EMBEDDER", "0")))

# ============ MODELS ============
# CRS system models to test
LLM_SYSTEMS = ["gemma:2b", "qwen:7b","qwen:4b", "llama3:instruct", "llama2:latest", "mistral:7b"]

# User simulator model
USER_MODEL = "llama3:instruct"

# LLM judge model
JUDGE_MODEL = "llama3:instruct"

# Topic extractor model (used if TOPIC_EXTRACTOR_MODE == "llm")
TOPIC_EXTRACTOR_MODEL = "mistral:7b"
TOPIC_EXTRACTOR_MODE = "lda"   # "llm", "heuristic", or "lda"

# ============ SIMULATION ============
N_TURNS = 20               # turns per conversation
ERROR_RATE = 0.3             # chance to start with wrong genreollama 

# ============ EVAL THRESHOLDS ============
# similarity below -> topic shift; above -> aligned
SIM_TOPIC_SHIFT = 0.55
ALIGNMENT_THRESHOLD = 0.65
TOPIC_JACCARD_SHIFT = 0.35

# ============ CAS WEIGHTS (sum doesn't have to be 1; code normalizes) ============
CAS_WEIGHTS = {
    "topic_recovery_rate": 0.01,
    "avg_recovery_delay": 0.02,        # will be used as (1 - normalized_delay)
    "topic_interference": 0.02,        # will be used as (1 - normalized_interference)
    "cross_coherence": 0.3,
    "context_retention": 0.15
}

# ============ PATHS ============
LOG_DIR = "logs"
RESULTS_FILE = "results.jsonl"
SUMMARY_CSV = "model_metrics.csv"
STAT_RESULTS_FILE = "statistical_analysis_results.txt"
FIG_DIR = "figures"
LDA_MODEL_DIR = "models"
LDA_MODEL_PATH = os.path.join(LDA_MODEL_DIR, "lda.model")
LDA_DICT_PATH = os.path.join(LDA_MODEL_DIR, "lda.dict")

# ============ EMBEDDINGS ============
# By default, use the fine-tuned model if it exists, otherwise fall back to the base model.
# Run `python finetune_embeddings.py` to create the fine-tuned model.
FINETUNED_EMBEDDING_PATH = "models/finetuned_embeddings"
EMBEDDING_MODEL_NAME = FINETUNED_EMBEDDING_PATH if os.path.exists(FINETUNED_EMBEDDING_PATH) else "all-MiniLM-L6-v2"

# ============ DEBUG / VERBOSITY ============
VERBOSE_EVAL = bool(int(os.getenv("VERBOSE_EVAL", "0")))
