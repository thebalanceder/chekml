# main.py

from config import (
    QUERY,
    TARGET_CSV_COUNT,
    MODELS,
    PROCESSED_DATASETS_FILE,
    METADATA_FOLDER,
    DOWNLOAD_FOLDER
)

from dataset_crawler import search_ckan_datasets, filter_datasets_with_csv
from dataset_evaluator import rank_datasets_parallel
from dataset_downloader import download_csv_resources
from utils import (
    load_processed_dataset_ids,
    save_processed_dataset_ids,
    save_json,
    ensure_folder_exists
)

# ============================================
# Save Dataset Metadata
# ============================================

def save_dataset_metadata(datasets, folder=METADATA_FOLDER, save_as_json=True, save_individual_files=True):
    ensure_folder_exists(folder)

    if save_as_json:
        json_path = os.path.join(folder, "top_datasets_metadata.json")
        save_json(json_path, datasets)
        print(f"✅ Saved combined metadata JSON: {json_path}")

    if save_individual_files:
        for idx, dataset in enumerate(datasets, 1):
            text_path = os.path.join(folder, f"dataset_{idx}_metadata.txt")
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(f"Dataset {idx}:\n")
                text_file.write(f"Title: {dataset.get('dataset_title', 'N/A')}\n")
                text_file.write(f"URL: {dataset.get('dataset_url', 'N/A')}\n")
                text_file.write(f"Score: {dataset.get('score', 'N/A')}\n\n")

                description = dataset.get('description', 'No description available.')
                text_file.write(f"Description:\n{description}\n\n")

                text_file.write("CSV Resources:\n")
                for res in dataset.get("csv_resources", []):
                    text_file.write(f"  - {res.get('url', 'No URL')}\n")

            print(f"✅ Saved metadata text file: {text_path}")


# ============================================
# Main Workflow
# ============================================

def main():
    print(f"🔍 Searching datasets with query: '{QUERY}'")
    datasets = search_ckan_datasets(query=QUERY, page_size=5, max_pages=1)

    if not datasets:
        print("⚠️ No datasets found.")
        return

    # Filter datasets (skip already processed)
    filtered_datasets = filter_datasets_with_csv(datasets)

    if not filtered_datasets:
        print("⚠️ No CSV datasets available for evaluation.")
        return

    # Evaluate datasets with multiple models
    ranked_datasets = rank_datasets_parallel(filtered_datasets)

    if not ranked_datasets:
        print("⚠️ No datasets ranked successfully.")
        return

    # Get top N datasets
    top_datasets = ranked_datasets[:TARGET_CSV_COUNT]

    # Save dataset metadata
    save_dataset_metadata(top_datasets)

    # Ask user about downloads
    number_input = input("\n💾 Enter the number of CSV files to download per dataset (or press Enter to skip): ").strip()

    if number_input.isdigit():
        number_per_dataset = int(number_input)
        download_csv_resources(
            datasets=top_datasets,
            download_folder=DOWNLOAD_FOLDER,
            number_per_dataset=number_per_dataset
        )
        print(f"\n✅ Downloaded {number_per_dataset} CSV files per dataset into '{DOWNLOAD_FOLDER}'!")
    else:
        print("⚠️ No valid number provided. Skipping downloads.")

    # Save processed dataset IDs
    processed_ids = load_processed_dataset_ids(PROCESSED_DATASETS_FILE)
    new_ids = {ds["dataset_id"] for ds in top_datasets}
    all_processed_ids = processed_ids.union(new_ids)

    save_processed_dataset_ids(PROCESSED_DATASETS_FILE, all_processed_ids)

    print(f"✅ Processed dataset IDs updated. Total processed: {len(all_processed_ids)}")


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    main()

