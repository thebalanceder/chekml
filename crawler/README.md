# How to setup and run
## 1. Install Ollama
- ollama.com/download
## 2. Pull a Model
```bash
ollama pull deepseek-r1:7b
```
## 3. Set Ollama Server's Port
- by default using script server uses port 11500
```bash
export OLLAMA_HOST=0.0.0.0:11500
```
## 4. Configuration of Script
Open SequentialCrawler.py in a text editor. Key configurable settings at the top:

`QUERY = "urban heat island"`: The search query for datasets (e.g., change to "climate change" for different topics).
`TARGET_CSV_COUNT = 1`: Max number of top-ranked datasets to process (after filtering for those with CSVs). Increase for more results.
`DOWNLOAD_FOLDER = "downloaded_datasets"`: Folder for downloaded CSVs (created automatically).
In `search_ckan_datasets` function:
`page_size=1`: Datasets per API page (increase for faster bulk fetching, e.g., 10).
`max_pages=1`: Max pages to fetch (increase for more results, e.g., 10 for up to 100 datasets if page_size=10).
In `ask_deepseek_about_geo` function:
`model="deepseek-r1:7b"`: Matches the pulled model.
`port=11500`: Matches the Ollama server port.
In `rank_dataset` function:
`deepseek_enabled=True`: Enable/disable LLM geo-scoring.
In `download_csv_resources` function:
`max_workers=5`: Max parallel download threads (increase for faster downloads, but beware of rate limits).
Other implicit settings:
`PROCESSED_DATASETS_FILE = "dataset_metadata/processed_datasets.json"`: Tracks processed dataset IDs.
Metadata saved to `dataset_metadata/` folder (JSON and optional TXT files).
