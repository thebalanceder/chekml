# dataset_crawler.py

import requests
from config import CKAN_API_URL, QUERY, PROCESSED_DATASETS_FILE
from utils import load_processed_dataset_ids

def search_ckan_datasets(query, page_size=5, max_pages=1):
    """
    Search datasets from CKAN API.
    """
    all_datasets = []
    start = 0

    for page in range(max_pages):
        params = {
            "q": query,
            "rows": page_size,
            "start": start
        }

        print(f"🔍 Searching datasets (page {page + 1}, start={start})...")
        try:
            response = requests.get(CKAN_API_URL, params=params, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❗ Request failed: {e}")
            break

        result = response.json().get("result", {})
        datasets = result.get("results", [])

        if not datasets:
            print("⚠️ No more datasets found.")
            break

        all_datasets.extend(datasets)
        start += page_size

    print(f"✅ Found {len(all_datasets)} datasets from CKAN.")
    return all_datasets


def filter_datasets_with_csv(datasets):
    """
    Filter datasets that:
    - Have CSV resources.
    - Haven't been processed yet.
    """
    processed_ids = load_processed_dataset_ids(PROCESSED_DATASETS_FILE)
    print(f"🔎 Skipping {len(processed_ids)} previously processed datasets.")

    filtered_datasets = []
    for dataset in datasets:
        dataset_id = dataset.get("id")
        dataset_title = dataset.get("title", "Unknown Title")

        # Skip already processed
        if dataset_id in processed_ids:
            print(f"⚠️ Skipping dataset already processed: {dataset_title}")
            continue

        # Filter for CSV resources
        csv_resources = [
            res for res in dataset.get("resources", [])
            if res.get("format", "").lower() == "csv" and res.get("url")
        ]

        if not csv_resources:
            print(f"⚠️ Skipping dataset with no CSV: {dataset_title}")
            continue

        # Keep dataset if it passed the filters
        filtered_datasets.append({
            "id": dataset_id,
            "title": dataset_title,
            "url": dataset.get("url", ""),
            "resources": csv_resources
        })

    print(f"✅ Filtered {len(filtered_datasets)} datasets with CSVs ready for evaluation.")
    return filtered_datasets

