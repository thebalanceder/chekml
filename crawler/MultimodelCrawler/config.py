# config.py

# ============================================
# CKAN Search Configuration
# ============================================
CKAN_API_URL = "https://catalog.data.gov/api/3/action/package_search"
QUERY = "urban heat island"
TARGET_CSV_COUNT = 2  # How many datasets to fetch/evaluate each run

# ============================================
# Dataset Storage Paths
# ============================================
PROCESSED_DATASETS_FILE = "dataset_metadata/processed_datasets.json"
METADATA_FOLDER = "dataset_metadata"
DOWNLOAD_FOLDER = "downloaded_datasets"

# ============================================
# Parallel Model Configurations
# ============================================

# Define the models you want to use
MODELS = [
    "deepseek-r1:7b",
    "tinyllama:1.1b"
]

# Number of instances for each model (can vary by model)
NUM_INSTANCES_PER_MODEL = {
    "deepseek-r1:7b": 1,
    "tinyllama:1.1b": 1
}

# Starting port for Ollama servers
OLLAMA_START_PORT = 11500

# This generates the ports allocated per model, based on NUM_INSTANCES_PER_MODEL
MODEL_PORT_MAP = {}
port_counter = OLLAMA_START_PORT

for model in MODELS:
    instance_count = NUM_INSTANCES_PER_MODEL[model]
    MODEL_PORT_MAP[model] = [port_counter + i for i in range(instance_count)]
    port_counter += instance_count

# ============================================
# Download Configuration
# ============================================
DEFAULT_MAX_DOWNLOAD_WORKERS = 5

# ============================================
# Debug & Logs
# ============================================
DEBUG_MODE = True  # Turn to False to minimize output later

