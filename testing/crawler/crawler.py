import requests

def get_keywords_with_ollama(prompt_text, model="deepseek-r1:7b"):
    url = "http://localhost:11500/api/generate"

    instruction = (
        f"List 10 dataset-related keywords that can be used to predict the Urban Heat Index (UHI).\n"
        "DO NOT provide any explanation or reasoning.\n"
        "ONLY output a numbered list.\n"
        "For example:\n"
        "1. Keyword 1\n"
        "2. Keyword 2\n"
        "...\n"
        "10. Keyword 10"
    )

    payload = {
        "model": model,
        "prompt": instruction,
        "stream": False
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()
    return data["response"].strip()

# Usage
query = "dataset to predict UHI Index"
keywords = get_keywords_with_ollama(query)

print("Suggested Keywords:")
print(keywords)

