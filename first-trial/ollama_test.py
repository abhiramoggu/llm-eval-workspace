# ollama_test.py
import subprocess

def call_ollama(model_name: str, prompt: str) -> str:
    """
    Call an Ollama model and stream its response line by line.
    """
    process = subprocess.Popen(
        ["ollama", "run", model_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Send the prompt
    output, error = process.communicate(input=prompt)

    if error:
        print("Error:", error)

    return output.strip()


if __name__ == "__main__":
    model = "gemma:2b"  # try a small model first
    prompt = "Recommend me a romantic movie."
    response = call_ollama(model, prompt)

    print(f"MODEL ({model}) RESPONSE:\n{response}")
