# dataset_downloader.py

import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import ensure_folder_exists

def download_csv_file(csv_url, dataset_idx, res_idx, download_folder):
    """
    Downloads a single CSV file and saves it to disk.
    """
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
    """
    Downloads CSV files from datasets using multithreading.
    """
    if number_per_dataset is None:
        print("⚠️ Download skipped: 'number' parameter not set.")
        return

    ensure_folder_exists(download_folder)
    download_tasks = []

    # Prepare download tasks
    for idx, dataset in enumerate(datasets, 1):
        csv_resources = dataset["csv_resources"]
        limited_resources = csv_resources[:number_per_dataset]

        for res_idx, resource in enumerate(limited_resources, 1):
            csv_url = resource["url"]
            download_tasks.append((csv_url, idx, res_idx))

    # Multithreaded download
    print(f"\n🚀 Starting multi-threaded download with {max_workers} workers...\n")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_csv_file, csv_url, dataset_idx, res_idx, download_folder): csv_url
            for csv_url, dataset_idx, res_idx in download_tasks
        }

        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"⚠️ Download failed for: {url}")
            except Exception as e:
                print(f"❗ Exception downloading {url}: {e}")

    print(f"\n✅ All downloads completed! Files saved in '{download_folder}'\n")

