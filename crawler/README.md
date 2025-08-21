# ALl Settings of Scripts
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
### Configuration of Script

## SequentialCrawler
Open `SequentialCrawler.py` in a text editor. Key configurable settings at the top:

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

 - Run:
   ```bash
   python3 SequentialCrawler.py
   ```
   
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
   - `NUM_MODELS = 2: Number of LLM instances for parallel evaluation.
   - `DEESEEK_START_PORT = 11500`: Starting port for LLM servers.
   - `DEESEEK_PORTS`: Auto-generated list [11500, 11501] (based on NUM_MODELS).
 - Other Implicit Settings:
   - In `dataset_crawler.py` (`search_ckan_datasets`):
     - `page_size=5`: Datasets per API call (increase for faster fetching, max ~1000).
     - `max_pages=1`: Pages to fetch (increase for more datasets, e.g., 10 for 50 datasets).
   - In `dataset_downloader.py` (download_csv_resources):
     - `max_workers=5`: Max download threads (adjust for CPU/network capacity).
   - In `dataset_evaluator.py`:
     - `traditional_score`: Weights: +5 for query keywords in title, +2 per CSV, +2 for datasets from 2020+, +2 for open licenses (-capped at 10).
     - `get_first_rows_from_csv`: `num_rows=5`, `max_cols=5`, `max_chars=2000` (limits CSV preview size for LLM).
     - Final score: 0.5 × traditional + 0.5 × LLM geo-score (capped at 10).
   - In `main.py` (`save_dataset_metadata`):
     - `save_as_json=True`: Saves combined JSON metadata.
     - `save_individual_files=True`: Saves per-dataset TXT files.

 - Run:
   ```bash
   python3 SequentialCrawler.py
   ```
   
## MutimodelCrawler
Edit config.py for all key settings:

 - CKAN Search:
 - `CKAN_API_URL = "https://catalog.data.gov/api/3/action/package_search"`: API endpoint (change for other CKAN sites).
 - `QUERY = "urban heat island"`: Search query (e.g., "air quality site:gov" for precision).
 - `TARGET_CSV_COUNT = 2`: Top-ranked datasets to process (increase for more).
 - Dataset Paths:
 - `PROCESSED_DATASETS_FILE = "dataset_metadata/processed_datasets.json"`: Tracks processed IDs.
 - `METADATA_FOLDER = "dataset_metadata"`: For JSON/TXT metadata.
 - `DOWNLOAD_FOLDER = "downloaded_datasets"`: For CSVs.
 - Model Configurations:
 - `MODELS = ["deepseek-r1:7b", "tinyllama:1.1b"]`: LLMs to cycle through (add/remove as needed; pull via Ollama first).
 - `NUM_INSTANCES_PER_MODEL = {"deepseek-r1:7b": 1, "tinyllama:1.1b": 1}`: Instances per model (increase for parallelism; requires more servers/ports).
 - `OLLAMA_START_PORT = 11500`: Starting port; MODEL_PORT_MAP auto-generates (e.g., `{ "deepseek-r1:7b": [11500], "tinyllama:1.1b": [11501] }`).
 - Download:
 - `DEFAULT_MAX_DOWNLOAD_WORKERS = 5`: Max threads for downloads (adjust for system/network).
 - Debug & Logs:
 - `DEBUG_MODE = True`: Enables debug prints (set to False for quieter output).

 - Run:
   ```bash
   python3 SequentialCrawler.py
   ```
   
## MixturemodelCrawler
Edit config.py for primary settings:

 - CKAN Search:
 - `CKAN_API_URL`: API endpoint.
 - `QUERY = "urban heat island"`: Search query.
 - `TARGET_CSV_COUNT = 5`: Top datasets to process.
 - Paths:
 - `PROCESSED_DATASETS_FILE`: Processed IDs JSON.
 - `METADATA_FOLDER`: Metadata output.
 - `DOWNLOAD_FOLDER`: CSVs output.
 - Models & Ollama:
 - `MODELS = ["deepseek-r1:7b", "tinyllama:1.1b"]`: Available LLMs (pull first).
 - `NUM_MODELS_TO_USE = 2`: Randomly select this many per run.
 - `NUM_INSTANCES_PER_MODEL`: Instances dict (default: 2 each; scales servers/ports).
 - `OLLAMA_START_PORT = 11500`: Start port; MODEL_PORT_MAP auto-generates (e.g., DeepSeek: [11500, 11501], TinyLlama: [11502, 11503]).
 - Download:
 - `DEFAULT_MAX_DOWNLOAD_WORKERS = 5`: Download threads.
 - Debug:
 - `DEBUG_MODE = True`: Enable debug logs.
 - Other tunables:
 - `dataset_crawler.py` (`search_ckan_datasets`): `page_size=10` (per call), `max_pages=1` (total pages).
 - `dataset_evaluator.py`: Traditional weights (+5 title, +2/CSV, +2 recency >=2020, +2 open license; cap 10). CSV preview: `num_rows=5`, `max_cols=5`, `max_chars=2000`. Score: 0.5 traditional + 0.5 LLM (cap 10).
 - `main.py` (save_dataset_metadata): save_as_json=True, save_individual_files=True.

 - Run:
   ```bash
   python3 SequentialCrawler.py
   ```
   
### Comparison of Script

 **Core Settings for all**
 - Dataset Source : CKAN API (data.gov)
 - Search Query : Configurable (QUERY)
 - CSV Filtering : Filters datasets with CSVs, skips processed
 - Metadata Saving : JSON + optional TXT files
 - CSV Downloads : Multithreaded, user-specified number per dataset
 - Processed Dataset Tracking : JSON file (processed_datasets.json)
 - Scoring Machanism : Title (+5), CSVs (+2 each), recency (≥2020, +2), open license (+2), capped at 10

| **Ranking Mechanism**       | SequentialCrawler | ParallelCrawler | MultiModelCrawler | MixtureModelCrawler |
|-----------------------------|-------------------|-----------------|-------------------|---------------------|
| Scoring Method              | Traditional + optional LLM score | Traditional + LLM score (async parallel) | Traditional + LLM score (async parallel, multiple models) | Traditional + LLM score (async parallel, random model subset) |
| LLM Geo-Scoring             | Single model (DeepSeek-R1:7b), optional | Single model (TinyLlama:1.1b or DeepSeek), parallel evaluation | Multiple models (DeepSeek, TinyLlama), cycles through all | Random subset of models (DeepSeek, TinyLlama), cycles through subset |
| LLM Parallelism             | None (sequential evaluation) | Multiple instances of one model (ports 11500+) | Multiple models, one instance each (ports 11500+) | Multiple models, multiple instances (ports 11500+) |

| **LLM Configuration**       | SequentialCrawler | ParallelCrawler | MultiModelCrawler | MixtureModelCrawler |
|-----------------------------|-------------------|-----------------|-------------------|---------------------|
| Default Model(s)            | DeepSeek-R1:7b | TinyLlama:1.1b (or DeepSeek-R1:7b) | DeepSeek-R1:7b, TinyLlama:1.1b | DeepSeek-R1:7b, TinyLlama:1.1b |
| Model Instances             | 1 (port 11500) | Configurable (NUM_MODELS, default 2, ports 11500–11501) | 1 per model (2 models, ports 11500–11501) | Configurable per model (default 2 each, ports 11500–11503) |
| Model Selection             | Fixed (single model) | Fixed (single model, multiple instances) | Fixed cycle through all models | Random subset (NUM_MODELS_TO_USE) each run |
| Disable LLM                 | `deepseek_enabled=False` | `model_geo_score=0.0` | `model_geo_score=0.0` | `llm_score=0.0` |

| **Settings (config.py)**    | SequentialCrawler | ParallelCrawler | MultiModelCrawler | MixtureModelCrawler |
|-----------------------------|-------------------|-----------------|-------------------|---------------------|
| CKAN_API_URL                | Yes | Yes | Yes | Yes |
| QUERY                       | Yes (e.g., "urban heat island") | Yes | Yes | Yes |
| TARGET_CSV_COUNT            | Yes (default 1) | Yes (default 2) | Yes (default 2) | Yes (default 5) |
| PROCESSED_DATASETS_FILE     | Yes | Yes | Yes | Yes |
| METADATA_FOLDER             | Yes | Yes | Yes | Yes |
| DOWNLOAD_FOLDER             | Yes | Yes | Yes | Yes |
| MODELS                      | N/A (hardcoded DeepSeek) | Single model name | List of models | List of models |
| NUM_MODELS                  | N/A | Number of instances (default 2) | Implicit (length of MODELS) | N/A |
| NUM_MODELS_TO_USE           | N/A | N/A | N/A | Yes (default 2) |
| NUM_INSTANCES_PER_MODEL     | N/A | N/A | N/A | Dict per model (default 2 each) |
| OLLAMA_START_PORT           | Hardcoded (11500) | Yes (default 11500) | Yes (default 11500) | Yes (default 11500) |
| MODEL_PORT_MAP              | N/A | Auto-generated (from NUM_MODELS) | Auto-generated (from MODELS) | Auto-generated (from NUM_INSTANCES_PER_MODEL) |
| DEFAULT_MAX_DOWNLOAD_WORKERS| Hardcoded (5) | Yes (default 5) | Yes (default 5) | Yes (default 5) |
| DEBUG_MODE                  | N/A | Yes (default True) | Yes (default True) | Yes (default True) |

| **Other Settings**          | SequentialCrawler | ParallelCrawler | MultiModelCrawler | MixtureModelCrawler |
|-----------------------------|-------------------|-----------------|-------------------|---------------------|
| page_size                   | Hardcoded (1) | Hardcoded (5) | Hardcoded (5) | Hardcoded (10) |
| max_pages                   | Hardcoded (1) | Hardcoded (1) | Hardcoded (1) | Hardcoded (1) |
| save_as_json                | Yes (default True) | Yes | Yes | Yes |
| save_individual_files       | Yes (default True) | Yes | Yes | Yes |
| CSV Preview                 | num_rows=5, max_cols=5, max_chars=2000 | Same | Same | Same |

| **Performance**             | SequentialCrawler | ParallelCrawler | MultiModelCrawler | MixtureModelCrawler |
|-----------------------------|-------------------|-----------------|-------------------|---------------------|
| Evaluation Speed            | Slowest (sequential) | Faster (parallel instances) | Balanced (multiple models) | Most flexible (random subset, multiple instances) |
| Scalability                 | Limited (single model) | Moderate (single model, multiple instances) | Good (multiple models) | Best (random model subset, multiple instances) |
| Resource Usage              | Low (~8GB RAM for DeepSeek) | Moderate (~4–8GB RAM per instance) | Moderate (~10GB for 2 models) | High (~20GB for 4 instances) |
| Use Case                    | Simple, low-resource systems; single model evaluation | Balanced performance; single model with parallelism | Diverse model evaluations | Flexible, high-throughput with varied model subset |
| Ollama Setup                | 1 server (port 11500) | 2 servers (11500–11501) | 2 servers (11500–11501) | 4 servers (11500–11503) |
