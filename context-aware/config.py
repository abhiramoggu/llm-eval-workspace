# config.py
# ============ MODES ============
# "mock" -> no LLM calls, simple placeholders
# "ollama" -> uses local Ollama models for USER / CRS / Judge / Topic extraction
MODE = "ollama"

# ============ MODELS ============
# CRS system models to test
LLM_SYSTEMS = ["gemma:2b"]
    #"gemma:2b", "mistral:7b", "llama3:instruct", ]

# User simulator model
USER_MODEL = "mistral:7b"

# LLM judge model
JUDGE_MODEL = "llama2:7b"

# Topic extractor model (used if TOPIC_EXTRACTOR_MODE == "llm")
TOPIC_EXTRACTOR_MODEL = "mistral:7b"
TOPIC_EXTRACTOR_MODE = "llm"   # "llm" or "heuristic"

# ============ SIMULATION ============
N_TURNS = 20               # turns per conversation
ERROR_RATE = 0.3             # chance to start with wrong genre

# ============ EVAL THRESHOLDS ============
# similarity below -> topic shift; above -> aligned
SIM_TOPIC_SHIFT = 0.55
ALIGNMENT_THRESHOLD = 0.65

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
