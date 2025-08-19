import requests
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

CKAN_API_URL = "https://catalog.data.gov/api/3/action/package_search"

# ----------------------------
# CONFIGURATION
# ----------------------------
QUERY = "urban heat island"
TARGET_CSV_COUNT = 10
DOWNLOAD_FOLDER = "downloaded_datasets"

# ----------------------------
# Dataset Search
# ----------------------------
def search_ckan_datasets(query, page_size=100, max_pages=10):
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
def rank_dataset(dataset, query_keywords):
    score = 0
    title = dataset.get("title", "").lower()

    # 1. Relevance to query
    if any(keyword.lower() in title for keyword in query_keywords):
        score += 5

    # 2. CSV resources count
    csv_count = sum(1 for res in dataset.get("resources", []) if res.get("format", "").lower() == "csv")
    score += csv_count * 2

    # 3. Recency of creation
    created_date = dataset.get("metadata_created", "")[:10]
    try:
        year = int(created_date[:4])
        if year >= 2020:
            score += 2
    except ValueError:
        pass

    # 4. License (Open license)
    license_id = (dataset.get("license_id") or "").lower()
    if license_id in ["cc-by", "cc-zero", "odc-pddl", "public-domain"]:
        score += 2

    return score

# ----------------------------
# Filter CSV Datasets + Rank
# ----------------------------
def filter_and_rank_datasets(datasets, query, target_csv_count):
    ranked_csv_datasets = []
    query_keywords = query.split()

    for dataset in datasets:
        # Check if there are CSV resources
        csv_resources = [
            res for res in dataset.get("resources", [])
            if res.get("format", "").lower() == "csv" and res.get("url")
        ]
        
        if not csv_resources:
            continue

        # Score the dataset
        score = rank_dataset(dataset, query_keywords)

        ranked_csv_datasets.append({
            "dataset_title": dataset["title"],
            "dataset_url": dataset.get("url", ""),
            "score": score,
            "csv_resources": csv_resources
        })

    # Sort by score
    ranked_csv_datasets.sort(key=lambda d: d["score"], reverse=True)

    # Limit to target count
    return ranked_csv_datasets[:target_csv_count]

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

    print(f"\n🎉 Top {len(ranked_csv_datasets)} CSV datasets:")
    for idx, dataset in enumerate(ranked_csv_datasets, 1):
        print(f"{idx}. Title: {dataset['dataset_title']}")
        print(f"   Dataset URL: {dataset['dataset_url']}")
        print(f"   Score: {dataset['score']}")
        print(f"   CSV Resources:")
        for res in dataset["csv_resources"]:
            print(f"     - {res['url']}")
        print()

    # Ask user for how many CSVs per dataset they want to download
    number_input = input("💾 Enter the number of CSV files to download per dataset (or press Enter to skip): ").strip()

    if number_input.isdigit():
        number_per_dataset = int(number_input)
        download_csv_resources(ranked_csv_datasets, DOWNLOAD_FOLDER, number_per_dataset)
        print(f"\n✅ Downloaded {number_per_dataset} CSV files per dataset into '{DOWNLOAD_FOLDER}'!")
    else:
        print("⚠️ No valid number provided. Skipping downloads.")


if __name__ == "__main__":
    main()

