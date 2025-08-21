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
### Configuration of Script

| Feature / Setting          | SequentialCrawler | ParallelCrawler | MultiModelCrawler | MixtureModelCrawler |
|-----------------------------|-------------------|-----------------|-------------------|---------------------|
| **Core Functionality**      |                   |                 |                   |                     |
| Dataset Source              | CKAN API (data.gov) | CKAN API (data.gov) | CKAN API (data.gov) | CKAN API (data.gov) |
| Search Query                | Configurable (QUERY) | Configurable (QUERY) | Configurable (QUERY) | Configurable (QUERY) |
| CSV Filtering               | Filters datasets with CSVs, skips processed | Same | Same | Same |
| Metadata Saving             | JSON + optional TXT files | Same | Same | Same |
| CSV Downloads               | Multithreaded, user-specified number per dataset | Same | Same | Same |
| Processed Dataset Tracking  | JSON file (processed_datasets.json) | Same | Same | Same |
| **Ranking Mechanism**       |                   |                 |                   |                     |
| Scoring Method              | Traditional + optional LLM geo-score | Traditional + LLM geo-score (async parallel) | Traditional + LLM geo-score (async parallel, multiple models) | Traditional + LLM geo-score (async parallel, random model subset) |
| Traditional Scoring         | Title (+5), CSVs (+2 each), recency (≥2020, +2), open license (+2), capped at 10 | Same | Same | Same |
| LLM Geo-Scoring             | Single model (DeepSeek-R1:7b), optional | Single model (TinyLlama:1.1b or DeepSeek), parallel evaluation | Multiple models (DeepSeek, TinyLlama), cycles through all | Random subset of models (DeepSeek, TinyLlama), cycles through subset |
| LLM Parallelism             | None (sequential evaluation) | Multiple instances of one model (ports 11500+) | Multiple models, one instance each (ports 11500+) | Multiple models, multiple instances (ports 11500+) |
| **LLM Configuration**       |                   |                 |                   |                     |
| Default Model(s)            | DeepSeek-R1:7b | TinyLlama:1.1b (or DeepSeek-R1:7b) | DeepSeek-R1:7b, TinyLlama:1.1b | DeepSeek-R1:7b, TinyLlama:1.1b |
| Model Instances             | 1 (port 11500) | Configurable (NUM_MODELS, default 2, ports 11500–11501) | 1 per model (2 models, ports 11500–11501) | Configurable per model (default 2 each, ports 11500–11503) |
| Model Selection             | Fixed (single model) | Fixed (single model, multiple instances) | Fixed cycle through all models | Random subset (NUM_MODELS_TO_USE) each run |
| Disable LLM                 | `deepseek_enabled=False` | `model_geo_score=0.0` | `model_geo_score=0.0` | `llm_score=0.0` |
| **Settings (config.py)**    |                   |                 |                   |                     |
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
| **Other Settings**          |                   |                 |                   |                     |
| page_size                   | Hardcoded (1) | Hardcoded (5) | Hardcoded (5) | Hardcoded (10) |
| max_pages                   | Hardcoded (1) | Hardcoded (1) | Hardcoded (1) | Hardcoded (1) |
| save_as_json                | Yes (default True) | Yes | Yes | Yes |
| save_individual_files       | Yes (default True) | Yes | Yes | Yes |
| CSV Preview                 | num_rows=5, max_cols=5, max_chars=2000 | Same | Same | Same |
| **Performance**             | 
| Evaluation Speed            | Slowest (sequential) | Faster (parallel instances) | Balanced (multiple models) | Most flexible (random subset, multiple instances) |
| Scalability                 | Limited (single model) | Moderate (single model, multiple instances) | Good (multiple models) | Best (random model subset, multiple instances) |
| Resource Usage              | Low (~8GB RAM for DeepSeek) | Moderate (~4–8GB RAM per instance) | Moderate (~10GB for 2 models) | High (~20GB for 4 instances) |
| Use Case                    | Simple, low-resource systems; single model evaluation | Balanced performance; single model with parallelism | Diverse model evaluations | Flexible, high-throughput with varied model subset |
| Ollama Setup                | 1 server (port 11500) | 2 servers (11500–11501) | 2 servers (11500–11501) | 4 servers (11500–11503) |
