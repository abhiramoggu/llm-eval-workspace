# config.py
# Global configuration for models and experiment settings

# "mock" for dry-run testing without LLMs
MODE = "ollama"

# CRS systems to test — these are the models acting as recommenders
LLM_SYSTEMS = ["gemma:2b", "mistral:7b"]#, "phi3:3b", "llama3:instruct"]

# Model that simulates the user
USER_MODEL = "mistral:7b"

# Model used for qualitative evaluation (LLM-as-judge)
JUDGE_MODEL = "llama2:7b"

# Probability that the user initially gives a wrong genre
ERROR_RATE = 0.6
