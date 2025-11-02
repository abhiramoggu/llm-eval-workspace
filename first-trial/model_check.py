# model_check.py
import subprocess

# List of models you want to test (replace/add your actual models)
MODELS_TO_TEST = [
    "gemma:2b",
    "gemma:7b",
    "qwen:4b",
    "qwen:7b",
    "deepseek:7b",
    "mistral:7b",
    "llama2:7b",
    "phi3:mini"
]

def call_ollama(model_name: str, prompt: str) -> str:
    """
    Test if a model runs successfully in Ollama.
    Returns the first line of the response or an error message.
    """
    try:
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=prompt, timeout=30)

        if error:
            return f"ERROR: {error.strip()}"
        if not output.strip():
            return "NO OUTPUT"
        return output.strip().split("\n")[0]  # return first line for brevity

    except subprocess.TimeoutExpired:
        return "TIMEOUT (took >30s)"
    except Exception as e:
        return f"FAILED: {str(e)}"


if __name__ == "__main__":
    print("🔍 Checking Ollama models...\n")
    prompt = "Say a one-sentence movie recommendation."
    results = {}

    for model in MODELS_TO_TEST:
        print(f"Testing {model}...")
        response = call_ollama(model, prompt)
        results[model] = response
        print(f"  → {response}\n")

    print("\n✅ Model Check Complete!")
    for model, res in results.items():
        print(f"{model}: {res}")
