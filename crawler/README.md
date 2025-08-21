# How to setup and run
- [SequentialCrawler](#SequentialCrawler)
- [ParallelCrawler](#ParallelCrawler)
- [MutimodelCrawler](#MutimodelCrawler)
- [MixturemodelCrawler](#MixturemodelCrawler)

### 1. Install Ollama
- ollama.com/download

### 2. Pull a Model
```bash
ollama pull deepseek-r1:7b
```
### 3. Set Ollama Server's Port
- by default using script server uses port 11500
```bash
export OLLAMA_HOST=0.0.0.0:11500
```
## SequentialCrawler

### Configuration of Script
Open SequentialCrawler.py in a text editor. Key configurable settings at the top:

 - `QUERY = "urban heat island"`: The search query for datasets (e.g., change to "climate change" for different topics).
 - `TARGET_CSV_COUNT = 1`: Max number of top-ranked datasets to process (after filtering for those with CSVs). Increase for more results.
 - `DOWNLOAD_FOLDER = "downloaded_datasets"`: Folder for downloaded CSVs (created automatically).
 - In `search_ckan_datasets` function:
   - `page_size=1`: Datasets per API page (increase for faster bulk fetching, e.g., 10).
   - `max_pages=1`: Max pages to fetch (increase for more results, e.g., 10 for up to 100 datasets if page_size=10).
 - In `ask_deepseek_about_geo` function:
   - `model="deepseek-r1:7b"`: Matches the pulled model.
   - `port=11500`: Matches the Ollama server port.
 - In `rank_dataset` function:
   - `deepseek_enabled=True`: Enable/disable LLM geo-scoring.
 - In `download_csv_resources` function:
   - `max_workers=5`: Max parallel download threads (increase for faster downloads, but beware of rate limits).
 - Other implicit settings:
   - `PROCESSED_DATASETS_FILE = "dataset_metadata/processed_datasets.json"`: Tracks processed dataset IDs.
   - Metadata saved to `dataset_metadata/` folder (JSON and optional TXT files).

 ## ParallelCrawler
 All settings are centralized in config.py. Edit as needed:

 - CKAN Settings:
   - `CKAN_API_URL = "https://catalog.data.gov/api/3/action/package_search"`: CKAN API endpoint (unchanged unless using a different CKAN instance).
   - `QUERY = "urban heat island"`: Search query (e.g., change to "air quality" for different datasets).
   - `TARGET_CSV_COUNT = 2`: Number of top-ranked datasets to process (increase for more results).
 - Dataset Paths:
   - `PROCESSED_DATASETS_FILE = "dataset_metadata/processed_datasets.json"`: Tracks processed dataset IDs.
   - `METADATA_FOLDER = "dataset_metadata"`: Stores JSON and TXT metadata.
   - `DOWNLOAD_FOLDER = "downloaded_datasets"`: Stores downloaded CSVs.
 - Model Settings:
   - `model_MODEL_NAME = "tinyllama:1.1b"`: LLM model (change to deepseek-r1:7b or custom).
   - `NUM_MODELS = 2`: Number of LLM instances for parallel evaluation.
   - `DEESEEK_START_PORT = 11500`: Starting port for LLM servers.
   - `DEESEEK_PORTS`: Auto-generated list `[11500, 11501]` (based on `NUM_MODELS`).
 - Other Implicit Settings:
   - In `dataset_crawler.py` (`search_ckan_datasets`):
     - `page_size=5`: Datasets per API call (increase for faster fetching, max ~1000).
     - `max_pages=1`: Pages to fetch (increase for more datasets, e.g., 10 for 50 datasets).
   - In `dataset_downloader.py` (download_csv_resources):
     - `max_workers=5`: Max download threads (adjust for CPU/network capacity).
   - In `dataset_evaluator.py`:
     - `traditional_score`: Weights: +5 for query keywords in title, +2 per CSV, +2 for datasets from 2020+, +2 for open licenses (capped at 10).
     - `get_first_rows_from_csv`: `num_rows=5`, `max_cols=5`, `max_chars=2000` (limits CSV preview size for LLM).
     - Final score: 0.5 × traditional + 0.5 × LLM geo-score (capped at 10).
   - In `main.py` (`save_dataset_metadata`):
     - `save_as_json=True`: Saves combined JSON metadata.
     - `save_individual_files=True`: Saves per-dataset TXT files.
 
 ## MutimodelCrawler
 ## MixturemodelCrawler
