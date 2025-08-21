# dataset_evaluator.py

import asyncio
import aiohttp
import pandas as pd
import re
from config import model_MODEL_NAME, DEESEEK_PORTS, QUERY
from utils import get_port_cycle

# ----------------------------
# Traditional Scoring Function
# ----------------------------

def traditional_score(dataset, query_keywords):
    score = 0
    title = dataset.get("title", "").lower()

    # Title relevance
    if any(keyword.lower() in title for keyword in query_keywords):
        score += 5

    # CSV resource count
    csv_count = sum(1 for res in dataset.get("resources", [])
                    if res.get("format", "").lower() == "csv")
    score += csv_count * 2

    # Dataset recency (created date)
    created_date = dataset.get("metadata_created", "")[:10]
    try:
        year = int(created_date[:4])
        if year >= 2020:
            score += 2
    except ValueError:
        pass

    # Open licenses boost
    license_id = (dataset.get("license_id") or "").lower()
    if license_id in ["cc-by", "cc-zero", "odc-pddl", "public-domain"]:
        score += 2

    return min(score, 10)  # Normalize to 10 max


# ----------------------------
# CSV Preview Reader
# ----------------------------

def get_first_rows_from_csv(csv_url, num_rows=5, max_cols=5, max_chars=2000):
    try:
        # Read fewer rows and fewer columns
        df = pd.read_csv(csv_url, nrows=num_rows, usecols=range(max_cols))
        csv_preview = df.to_csv(index=False)

        # Limit total character length of prompt (Ollama-friendly)
        if len(csv_preview) > max_chars:
            csv_preview = csv_preview[:max_chars]

        return csv_preview

    except Exception as e:
        print(f"❗ Failed to read CSV from {csv_url}: {e}")
        return None

# ----------------------------
# Async model Evaluator
# ----------------------------

async def ask_model_about_geo_async(session, csv_preview, port):
    url = f"http://localhost:{port}/api/generate"

    prompt = (
        "You are an AI evaluator. Analyze the following CSV preview and rate score 0 to 10 if the csv contain geo data. "
        "Geo data such as latitude, longitude, coordinates, or spatial identifiers. "
        "Return a single number only.\n\n"
        f"CSV Preview:\n{csv_preview}\n\n"
        "Score:"
    )
    
    """You are a helpful AI data quality evaluator. Analyze the following CSV data preview. "
    "Return ONLY a single number from 0 to 10 representing whether this dataset contains geo data. "
    "Geo data includes columns like latitude, longitude, coordinates, or spatial identifiers.\n\n"
    "❗ IMPORTANT: Do NOT provide explanations. Do NOT include <think> tags. "
    "ONLY return a number.\n\n"
    f"CSV Preview:\n{csv_preview}\n\n"
    "Your score (number only):"""

    payload = {
        "model": model_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        async with session.post(url, json=payload, timeout=60) as response:
            data = await response.json()
            raw_result = data.get("response", "").strip()
            print(f"🔎 Raw LLM Response (Port {port}): {raw_result}")

            match = re.search(r"[-+]?\d*\.\d+|\d+", raw_result)
            if match:
                score = float(match.group(0))
                score = max(0.0, min(score, 10.0))  # Cap between 0-10
                print(csv_preview)
                print(f"✅ model geo score extracted (Port {port}): {score}")
                return score
            else:
                print(f"❗ No number found in model response on port {port}.")
                return 0.0

    except Exception as e:
        print(f"❗ Failed model request on port {port}: {e}")
        return 0.0


# ----------------------------
# Dataset Evaluation Wrapper (Async)
# ----------------------------

async def evaluate_dataset(dataset, query_keywords, port, session):
    traditional = traditional_score(dataset, query_keywords)
    model_geo_score = 0.0

    first_csv_url = next(
        (res.get("url") for res in dataset.get("resources", [])
         if res.get("format", "").lower() == "csv"), None
    )

    if first_csv_url:
        csv_preview = get_first_rows_from_csv(first_csv_url)
        if csv_preview:
            model_geo_score = await ask_model_about_geo_async(session, csv_preview, port)

    # Combine scores (50/50)
    final_score = (traditional * 0.5) + (model_geo_score * 0.5)
    final_score = min(final_score, 10.0)

    print(f"⭐ Final dataset score (Port {port}): {final_score:.2f} "
          f"(Traditional: {traditional}, model: {model_geo_score})")

    return {
        "dataset_id": dataset["id"],
        "dataset_title": dataset["title"],
        "dataset_url": dataset.get("url", ""),
        "score": final_score,
        "csv_resources": dataset["resources"]
    }


# ----------------------------
# Parallel Evaluator Manager
# ----------------------------

async def evaluate_datasets_parallel(datasets, num_models=4):
    query_keywords = QUERY.split()
    port_cycle = get_port_cycle(DEESEEK_PORTS)

    tasks = []
    async with aiohttp.ClientSession() as session:
        for dataset in datasets:
            port = next(port_cycle)
            tasks.append(evaluate_dataset(dataset, query_keywords, port, session))

        results = await asyncio.gather(*tasks)

    return results


def rank_datasets_parallel(datasets, num_models=4):
    """
    Run parallel ranking for datasets and return sorted top datasets.
    """
    print(f"🚀 Evaluating {len(datasets)} datasets across {num_models} model models...")
    results = asyncio.run(evaluate_datasets_parallel(datasets, num_models=num_models))

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

