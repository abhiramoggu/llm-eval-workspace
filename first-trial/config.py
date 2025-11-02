# config.py

# Switch modes: "mock" or "ollama"
MODE = "ollama"

# CRS system models to test
LLM_SYSTEMS = ["gemma:2b"]
# "qwen:4b", "llama2:7b", "mistral:7b"]  # extend with ["gemma:2b", "qwen:4b", "deepseek:7b"]

# User simulator model
USER_MODEL = "llama3:instruct"

# LLM judge model
JUDGE_MODEL = "llama2:7b"

# Error injection probability (0–1)
ERROR_RATE = 0.7
