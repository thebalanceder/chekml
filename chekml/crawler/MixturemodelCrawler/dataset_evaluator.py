# dataset_evaluator.py

import asyncio
import aiohttp
import pandas as pd
import re

from config import MODELS, NUM_MODELS_TO_USE
from utils import (
    select_random_models,
    get_model_cycle,
    get_port_cycle_for_model,
    debug_log
)

# ============================================
# Traditional Scoring Function
# ============================================

def traditional_score(dataset, query_keywords):
    """
    Perform traditional scoring based on dataset metadata.
    """
    score = 0
    title = dataset.get("title", "").lower()

    # Title relevance
    if any(keyword.lower() in title for keyword in query_keywords):
        score += 5

    # CSV resource count
    csv_count = sum(1 for res in dataset.get("resources", []) if res.get("format", "").lower() == "csv")
    score += csv_count * 2

    # Dataset recency
    created_date = dataset.get("metadata_created", "")[:10]
    try:
        year = int(created_date[:4])
        if year >= 2020:
            score += 2
    except ValueError:
        pass

    # License openness
    license_id = (dataset.get("license_id") or "").lower()
    if license_id in ["cc-by", "cc-zero", "odc-pddl", "public-domain"]:
        score += 2

    return min(score, 10)  # Clamp score to 10


# ============================================
# CSV Preview Reader (limit rows, cols, size)
# ============================================

def get_first_rows_from_csv(csv_url, num_rows=5, max_cols=5, max_chars=2000):
    """
    Fetch and trim CSV preview for LLM evaluation.
    """
    try:
        df = pd.read_csv(csv_url, nrows=num_rows, usecols=range(max_cols))
        csv_preview = df.to_csv(index=False)

        if len(csv_preview) > max_chars:
            csv_preview = csv_preview[:max_chars]

        return csv_preview

    except Exception as e:
        print(f"❗ Failed to read CSV from {csv_url}: {e}")
        return None


# ============================================
# Async LLM Evaluator for Geo Data Detection
# ============================================

async def ask_model_about_geo_async(session, csv_preview, model_name, port):
    """
    Send CSV preview to a specific model on a specific port via Ollama API.
    """
    url = f"http://localhost:{port}/api/generate"

    prompt = (
        "You are an AI evaluator. Analyze the following CSV preview and return a number from 0 to 10. "
        "The score represents if the dataset includes geo data such as latitude, longitude, coordinates, or spatial identifiers. "
        "Return a single number only.\n\n"
        f"CSV Preview:\n{csv_preview}\n\n"
        "Score:"
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    debug_log(f"🚀 Sending request to {model_name} on port {port}...")

    try:
        async with session.post(url, json=payload, timeout=60) as response:
            data = await response.json()
            raw_result = data.get("response", "").strip()

            debug_log(f"🔎 Raw LLM Response ({model_name}:{port}): {raw_result}")

            match = re.search(r"[-+]?\d*\.\d+|\d+", raw_result)
            if match:
                score = float(match.group(0))

                # Clamp score between 0 and 10
                score = max(0.0, min(score, 10.0))

                if score > 10 or score < 0:
                    debug_log(f"⚠️ Out-of-bounds score ({score}) from {model_name}:{port}")

                debug_log(f"✅ LLM score extracted ({model_name}:{port}): {score}")
                return score

            debug_log(f"❗ No valid number found in response ({model_name}:{port}): {raw_result}")
            return 0.0

    except Exception as e:
        print(f"❗ Failed LLM request to {model_name}:{port} -> {e}")
        return 0.0


# ============================================
# Dataset Evaluation Wrapper (Single Dataset)
# ============================================

async def evaluate_dataset(dataset, query_keywords, model_name, port, session):
    """
    Evaluate a single dataset with traditional scoring + LLM scoring.
    """
    traditional = traditional_score(dataset, query_keywords)
    llm_score = 0.0

    first_csv_url = next(
        (res.get("url") for res in dataset.get("resources", [])
         if res.get("format", "").lower() == "csv"), None
    )

    if first_csv_url:
        csv_preview = get_first_rows_from_csv(first_csv_url)
        if csv_preview:
            llm_score = await ask_model_about_geo_async(session, csv_preview, model_name, port)

    # Final blended score
    final_score = (traditional * 0.5) + (llm_score * 0.5)
    final_score = min(final_score, 10.0)

    print(f"⭐ Final dataset score ({model_name}:{port}): {final_score:.2f} "
          f"(Traditional: {traditional}, {model_name}: {llm_score})")

    return {
        "dataset_id": dataset["id"],
        "dataset_title": dataset["title"],
        "dataset_url": dataset.get("url", ""),
        "score": final_score,
        "csv_resources": dataset["resources"]
    }


# ============================================
# Parallel Evaluator Manager (Multiple Datasets)
# ============================================

async def evaluate_datasets_parallel(datasets, query_keywords, selected_models):
    """
    Evaluate datasets in parallel using a random selection of models and their ports.
    """
    model_cycle = get_model_cycle(selected_models)
    tasks = []

    async with aiohttp.ClientSession() as session:
        for dataset in datasets:
            model_name = next(model_cycle)
            port_cycle = get_port_cycle_for_model(model_name)
            port = next(port_cycle)

            tasks.append(evaluate_dataset(dataset, query_keywords, model_name, port, session))

        results = await asyncio.gather(*tasks)

    return results


# ============================================
# Ranking Datasets (Entry Point)
# ============================================

def rank_datasets_parallel(datasets, query_keywords):
    """
    Randomly select N models, then rank datasets in parallel.
    """
    print(f"🎲 Selecting {NUM_MODELS_TO_USE} models from available models: {MODELS}")

    selected_models = select_random_models(MODELS, NUM_MODELS_TO_USE)

    print(f"🚀 Evaluating {len(datasets)} datasets using models: {selected_models}")

    results = asyncio.run(evaluate_datasets_parallel(datasets, query_keywords, selected_models))

    # Sort datasets by final score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"✅ Ranking complete! Top dataset score: {results[0]['score'] if results else 'N/A'}")
    return results

