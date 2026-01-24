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


# ============ EVALUATION MODE ============
# "grounded" -> catalog-grounded concept extraction + TF-IDF/cosine + Jaccard + n-gram copy penalty
# "legacy"   -> older topic-extraction experiments (heuristic/LDA/LLM)
EVAL_MODE = os.getenv("EVAL_MODE", "grounded")

# ============ CONCEPT/TOPIC FIELDS ============
# Canonical field list for grounded extraction (concepts); keep TOPIC_FIELDS as legacy alias.
TITLE_FIELD = "name"      # excluded from TAS by default; used for rec proxy
PLOT_FIELD = "plot_kw"    # excluded from TAS by default; appendix-only
USE_PLOT_KW_FOR_TAS = bool(int(os.getenv("USE_PLOT_KW_FOR_TAS", "0")))
USE_NAME_FOR_TAS = bool(int(os.getenv("USE_NAME_FOR_TAS", "0")))

STRUCTURED_CONCEPT_FIELDS = (
    "genre",
    "actor",
    "director",
    "writer",
    "language",
    "year",
)
CONCEPT_FIELDS = STRUCTURED_CONCEPT_FIELDS + (
    (TITLE_FIELD,) if USE_NAME_FOR_TAS else ()
) + (
    (PLOT_FIELD,) if USE_PLOT_KW_FOR_TAS else ()
)
TOPIC_FIELDS = CONCEPT_FIELDS

# Segment/recovery settings (used by grounded evaluation)
RECOVERY_WINDOW = int(os.getenv("RECOVERY_WINDOW", "3"))

# TAS weights (TAS = alpha*CC + beta*CR - gamma*I)
TAS_WEIGHTS = {
    "alpha": float(os.getenv("TAS_ALPHA", "1.0")),
    "beta": float(os.getenv("TAS_BETA", "1.0")),
    "gamma": float(os.getenv("TAS_GAMMA", "1.0")),
}

TOPIC_EXTRACTOR_MODEL = "mistral:7b"
CONCEPT_EXTRACTOR_MODEL = os.getenv("CONCEPT_EXTRACTOR_MODEL", TOPIC_EXTRACTOR_MODEL)
# NOTE: In the thesis/docs we use 'concept' (field=value grounded constraints) instead of 'topic'.
# Aliases below preserve backward-compatibility with earlier naming.

TOPIC_EXTRACTOR_MODE = os.getenv("TOPIC_EXTRACTOR_MODE", "lda")   # "llm", "heuristic", or "lda"
# Concept extractor mode for TAS/grounded evaluation: "catalog" or "llm".
CONCEPT_EXTRACTOR_MODE = os.getenv("CONCEPT_EXTRACTOR_MODE", "catalog")
CONCEPT_EMBED_THRESHOLD = float(os.getenv("CONCEPT_EMBED_THRESHOLD", "0.35"))
CONCEPT_EMBED_TOP_K = int(os.getenv("CONCEPT_EMBED_TOP_K", "3"))


# ============ SIMULATION ============
N_TURNS = int(os.getenv("N_TURNS", "50"))            # turns per conversation
N_SESSIONS = int(os.getenv("N_SESSIONS", "100"))     # total conversations per model
SHIFT_AFTER_TURNS = int(os.getenv("SHIFT_AFTER_TURNS", "12"))  # force first shift after this turn
ERROR_RATE = 0.3             # chance to start with wrong genreollama 

# ============ EVAL THRESHOLDS ============
# similarity below -> topic shift; above -> aligned
SIM_TOPIC_SHIFT = 0.55
ALIGNMENT_THRESHOLD = 0.65
TOPIC_JACCARD_SHIFT = 0.35
SIM_CONCEPT_SHIFT = SIM_TOPIC_SHIFT  # alias
CONCEPT_JACCARD_SHIFT = TOPIC_JACCARD_SHIFT  # alias

# Recommendation satisfaction proxy (disabled by default)
ENABLE_REC_SAT = bool(int(os.getenv("ENABLE_REC_SAT", "0")))
CONCEPT_ALIGNMENT_THRESHOLD = ALIGNMENT_THRESHOLD  # alias

# ============ CAS WEIGHTS (sum doesn't have to be 1; code normalizes) ============
ENABLE_CAS = False  # Keep recovery metrics, but do not combine them into a second headline score by default.

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
FIG_DIR_BASE = "figures"
_topic_mode_dir = TOPIC_EXTRACTOR_MODE.replace(":", "_")
FIG_DIR = os.path.join(FIG_DIR_BASE, _topic_mode_dir)
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
