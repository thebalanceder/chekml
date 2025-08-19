import requests
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import re
import json

CKAN_API_URL = "https://catalog.data.gov/api/3/action/package_search"
PROCESSED_DATASETS_FILE = "dataset_metadata/processed_datasets.json"

# ----------------------------
# CONFIGURATION
# ----------------------------
QUERY = "urban heat island"
TARGET_CSV_COUNT = 1
DOWNLOAD_FOLDER = "downloaded_datasets"

# ----------------------------
# Dataset Search
# ----------------------------

def load_processed_dataset_ids():
    if os.path.exists(PROCESSED_DATASETS_FILE):
        with open(PROCESSED_DATASETS_FILE, "r") as f:
            return set(json.load(f))
    else:
        return set()

def save_processed_dataset_ids(processed_ids):
    os.makedirs(os.path.dirname(PROCESSED_DATASETS_FILE), exist_ok=True)
    with open(PROCESSED_DATASETS_FILE, "w") as f:
        json.dump(list(processed_ids), f, indent=4)

def save_dataset_metadata(datasets, folder="dataset_metadata", save_as_json=True, save_individual_files=False):
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save all metadata in a single JSON file
    if save_as_json:
        json_path = os.path.join(folder, "top_datasets_metadata.json")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(datasets, json_file, indent=4)
        print(f"✅ All dataset metadata saved as JSON: {json_path}")

    # Optionally save each dataset's metadata in separate text files
    if save_individual_files:
        for idx, dataset in enumerate(datasets, 1):
            text_path = os.path.join(folder, f"dataset_{idx}_metadata.txt")
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(f"Dataset {idx}:\n")
                text_file.write(f"Title: {dataset.get('dataset_title', 'N/A')}\n")
                text_file.write(f"URL: {dataset.get('dataset_url', 'N/A')}\n")
                text_file.write(f"Score: {dataset.get('score', 'N/A')}\n\n")

                # Optional description if it exists (you may need to fetch this from CKAN if not included)
                description = dataset.get('description', 'No description available.')
                text_file.write(f"Description:\n{description}\n\n")

                # List CSV resources
                text_file.write("CSV Resources:\n")
                for res in dataset.get("csv_resources", []):
                    text_file.write(f"  - {res.get('url', 'No URL')}\n")

            print(f"✅ Saved individual metadata file: {text_path}")

def search_ckan_datasets(query, page_size=1, max_pages=1):
    all_datasets = []
    start = 0
    for _ in range(max_pages):
        params = {
            "q": query,
            "rows": page_size,
            "start": start
        }
        print(f"Searching datasets (start={start})...")
        response = requests.get(CKAN_API_URL, params=params)
        response.raise_for_status()

        result = response.json().get("result", {})
        datasets = result.get("results", [])
        if not datasets:
            break

        all_datasets.extend(datasets)
        start += page_size
    return all_datasets

# ----------------------------
# Ranking Function
# ----------------------------
def get_first_rows_from_csv(csv_url, num_rows=5):
    try:
        df = pd.read_csv(csv_url, nrows=num_rows)
        # Convert to CSV string (for LLM context)
        csv_preview = df.to_csv(index=False)
        return csv_preview
    except Exception as e:
        print(f"❗ Failed to read CSV {csv_url}: {e}")
        return None

def ask_deepseek_about_geo(csv_preview, model="deepseek-r1:7b", port=11500):
    url = f"http://localhost:{port}/api/generate"

    prompt = (
        "You are a helpful AI data quality evaluator. Analyze the following CSV data preview. "
        "Return ONLY a single number from 0 to 10 representing whether this dataset contains geo data. "
        "Geo data includes columns like latitude, longitude, coordinates, or spatial identifiers.\n\n"
        "❗ IMPORTANT: Do NOT provide explanations. Do NOT include <think> tags. "
        "ONLY return a number.\n\n"
        f"CSV Preview:\n{csv_preview}\n\n"
        "Your score (number only):"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        raw_result = response.json()["response"].strip()
        print(f"🔎 Raw LLM Response: {raw_result}")

        # Extract first number using regex
        match = re.search(r"[-+]?\d*\.\d+|\d+", raw_result)
        if match:
            score = float(match.group(0))
            print(f"✅ DeepSeek geo score extracted: {score}")
            return score
        else:
            print(f"❗ No number found in DeepSeek response: {raw_result}")
            return 0.0

    except Exception as e:
        print(f"❗ Failed to get score from DeepSeek: {e}")
        return 0.0

def rank_dataset(dataset, query_keywords, deepseek_enabled=True):
    # Traditional score (max ~12 based on your earlier scoring logic)
    base_score = 0
    title = dataset.get("title", "").lower()

    if any(keyword.lower() in title for keyword in query_keywords):
        base_score += 5

    csv_count = sum(1 for res in dataset.get("resources", []) if res.get("format", "").lower() == "csv")
    base_score += csv_count * 2

    created_date = dataset.get("metadata_created", "")[:10]
    try:
        year = int(created_date[:4])
        if year >= 2020:
            base_score += 2
    except ValueError:
        pass

    license_id = (dataset.get("license_id") or "").lower()
    if license_id in ["cc-by", "cc-zero", "odc-pddl", "public-domain"]:
        base_score += 2

    # Normalize base_score to a 10-point scale
    traditional_score = min(base_score, 10)

    # Ask DeepSeek about geo data (only if enabled and CSVs exist)
    deepseek_geo_score = 0.0
    if deepseek_enabled and csv_count > 0:
        first_csv_url = next(
            (res["url"] for res in dataset["resources"] if res.get("format", "").lower() == "csv"), None
        )

        if first_csv_url:
            csv_preview = get_first_rows_from_csv(first_csv_url)
            if csv_preview:
                deepseek_geo_score = ask_deepseek_about_geo(csv_preview)

    # Final score combines both scores (weighted 50/50)
    final_score = (traditional_score * 0.5) + (deepseek_geo_score * 0.5)

    print(f"⭐ Final dataset score: {final_score:.2f} (Traditional: {traditional_score}, DeepSeek: {deepseek_geo_score})")

    return final_score


# ----------------------------
# Filter CSV Datasets + Rank
# ----------------------------
def filter_and_rank_datasets(datasets, query, target_csv_count, deepseek_enabled=True):
    processed_ids = load_processed_dataset_ids()
    print(f"🔎 Skipping {len(processed_ids)} previously processed datasets.")

    ranked_csv_datasets = []
    query_keywords = query.split()

    for dataset in datasets:
        dataset_id = dataset.get("id")

        # Skip if dataset was already processed
        if dataset_id in processed_ids:
            print(f"⚠️ Skipping previously processed dataset: {dataset.get('title', 'Unknown')}")
            continue

        # Filter CSV resources
        csv_resources = [
            res for res in dataset.get("resources", [])
            if res.get("format", "").lower() == "csv" and res.get("url")
        ]

        if not csv_resources:
            continue

        # Rank dataset
        score = rank_dataset(dataset, query_keywords, deepseek_enabled=deepseek_enabled)

        ranked_csv_datasets.append({
            "dataset_id": dataset_id,
            "dataset_title": dataset["title"],
            "dataset_url": dataset.get("url", ""),
            "score": score,
            "csv_resources": csv_resources
        })

        # Stop if we hit our target
        if len(ranked_csv_datasets) >= target_csv_count:
            break

    ranked_csv_datasets.sort(key=lambda d: d["score"], reverse=True)
    return ranked_csv_datasets

# ----------------------------
# Download CSV Files (Optional)
# ----------------------------
def download_csv_file(csv_url, dataset_idx, res_idx, download_folder):
    """Single CSV downloader function (to be run in a thread)."""
    filename = f"{download_folder}/dataset_{dataset_idx}_{res_idx}.csv"

    print(f"⬇️ Downloading {csv_url} ...")
    try:
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"✅ Saved to {filename}")
        return True
    except Exception as e:
        print(f"❗ Failed to download {csv_url}: {e}")
        return False

def download_csv_resources(datasets, download_folder, number_per_dataset=None, max_workers=5):
    if number_per_dataset is None:
        print("⚠️ Download skipped: 'number' parameter not set.")
        return

    os.makedirs(download_folder, exist_ok=True)
    download_tasks = []

    # Collect download tasks
    for idx, dataset in enumerate(datasets, 1):
        csv_resources = dataset["csv_resources"]

        # Limit the number of CSV files per dataset
        limited_csv_resources = csv_resources[:number_per_dataset]

        for res_idx, resource in enumerate(limited_csv_resources, 1):
            csv_url = resource["url"]

            # Append task details
            download_tasks.append((csv_url, idx, res_idx))

    # Run downloads in parallel threads
    print(f"\n🚀 Starting multi-threaded download with {max_workers} workers...\n")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare future tasks
        future_to_url = {
            executor.submit(download_csv_file, csv_url, dataset_idx, res_idx, download_folder): csv_url
            for csv_url, dataset_idx, res_idx in download_tasks
        }

        # As tasks complete, print status
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if not result:
                    print(f"⚠️ Download failed for: {url}")
            except Exception as e:
                print(f"❗ Exception during download of {url}: {e}")

    print(f"\n✅ All downloads finished! Files are in '{download_folder}'\n")

# ----------------------------
# Main Process
# ----------------------------
def main():
    print(f"🔍 Searching for datasets related to: '{QUERY}'")
    datasets = search_ckan_datasets(QUERY)

    print(f"✅ {len(datasets)} datasets found. Filtering CSV datasets...")
    ranked_csv_datasets = filter_and_rank_datasets(datasets, QUERY, TARGET_CSV_COUNT)

    if not ranked_csv_datasets:
        print("⚠️ No new datasets to process.")
        return

    print(f"\n🎉 Top {len(ranked_csv_datasets)} CSV datasets:")
    for idx, dataset in enumerate(ranked_csv_datasets, 1):
        print(f"{idx}. Title: {dataset['dataset_title']}")
        print(f"   Dataset URL: {dataset['dataset_url']}")
        print(f"   Score: {dataset['score']}")
        print(f"   CSV Resources:")
        for res in dataset["csv_resources"]:
            print(f"     - {res['url']}")
        print()

    # Save descriptions/metadata
    save_dataset_metadata(
        datasets=ranked_csv_datasets,
        folder="dataset_metadata",              # Folder where files are saved
        save_as_json=True,                      # Save all metadata as one JSON file
        save_individual_files=True              # Save individual text files for each dataset
    )
    # Ask user for how many CSVs per dataset they want to download
    number_input = input("💾 Enter the number of CSV files to download per dataset (or press Enter to skip): ").strip()

    if number_input.isdigit():
        number_per_dataset = int(number_input)
        download_csv_resources(ranked_csv_datasets, DOWNLOAD_FOLDER, number_per_dataset)
        print(f"\n✅ Downloaded {number_per_dataset} CSV files per dataset into '{DOWNLOAD_FOLDER}'!")
    else:
        print("⚠️ No valid number provided. Skipping downloads.")

    # Save processed dataset IDs after this run
    processed_ids = load_processed_dataset_ids()
    new_ids = {ds["dataset_id"] for ds in ranked_csv_datasets}
    all_processed_ids = processed_ids.union(new_ids)
    save_processed_dataset_ids(all_processed_ids)

if __name__ == "__main__":
    main()
