# config.py

# CKAN Settings
CKAN_API_URL = "https://catalog.data.gov/api/3/action/package_search"
QUERY = "urban heat island"
TARGET_CSV_COUNT = 2  # How many datasets to rank per run

# Dataset Paths
PROCESSED_DATASETS_FILE = "dataset_metadata/processed_datasets.json"
METADATA_FOLDER = "dataset_metadata"
DOWNLOAD_FOLDER = "downloaded_datasets"

# model / Ollama Model Settings
model_MODEL_NAME = "tinyllama:1.1b"#"deepseek-r1:7b"

# Parallel Model Serving Settings
NUM_MODELS = 2
DEESEEK_START_PORT = 11500

# Generate list of ports for multiple model instances
DEESEEK_PORTS = [DEESEEK_START_PORT + i for i in range(NUM_MODELS)]

