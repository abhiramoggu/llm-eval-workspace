# config.py
import os

# ============ MODES ============
# "mock" -> no LLM calls, simple placeholders
# "ollama" -> uses local Ollama models for USER / CRS / Judge / Topic extraction
MODE = os.getenv("SIM_MODE", "ollama")

# Fail safe: alert if MODE differs from supported values
if MODE not in {"ollama", "mock"}:
    raise ValueError(f"Unsupported SIM_MODE '{MODE}'. Use 'ollama' or 'mock'.")

# ============ MODELS ============
# CRS system models to test
LLM_SYSTEMS = ["gemma:2b", "qwen:7b", "qwen:4b", "llama3:instruct", "llama2:latest", "mistral:7b"]

# User simulator model
USER_MODEL = "mistral:7b"

# LLM judge model
JUDGE_MODEL = "llama3:instruct"

# Topic extractor model (used if TOPIC_EXTRACTOR_MODE == "llm")
TOPIC_EXTRACTOR_MODEL = "mistral:7b"
TOPIC_EXTRACTOR_MODE = "llm"   # "llm" or "heuristic"

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
    "topic_recovery_rate": 0.25,
    "avg_recovery_delay": 0.20,        # will be used as (1 - normalized_delay)
    "topic_interference": 0.20,        # will be used as (1 - normalized_interference)
    "cross_coherence": 0.20,
    "context_retention": 0.15
}

# ============ PATHS ============
LOG_DIR = "logs"
RESULTS_FILE = "results.jsonl"
SUMMARY_CSV = "model_metrics.csv"
FIG_DIR = "figures"

# ============ EMBEDDINGS ============
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers

# ============ DEBUG / VERBOSITY ============
VERBOSE_EVAL = bool(int(os.getenv("VERBOSE_EVAL", "0")))
