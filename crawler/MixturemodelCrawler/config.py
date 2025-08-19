# config.py

# ============================================
# CKAN Search Configuration
# ============================================
CKAN_API_URL = "https://catalog.data.gov/api/3/action/package_search"
QUERY = "urban heat island"
TARGET_CSV_COUNT = 5  # Number of datasets to fetch/rank per run

# ============================================
# Dataset Storage Paths
# ============================================
PROCESSED_DATASETS_FILE = "dataset_metadata/processed_datasets.json"
METADATA_FOLDER = "dataset_metadata"
DOWNLOAD_FOLDER = "downloaded_datasets"

# ============================================
# Models & Ollama Configuration
# ============================================

# The complete list of available models
MODELS = [
    "deepseek-r1:7b",
    "tinyllama:1.1b"
]

# How many models to randomly select each run
NUM_MODELS_TO_USE = 2  # Set N here

# Number of server instances per model (adjust if you want more servers per model)
NUM_INSTANCES_PER_MODEL = {
    "deepseek-r1:7b": 2,
    "tinyllama:1.1b": 2
}

# Ollama server port allocation
OLLAMA_START_PORT = 11500

# Automatically assign ports based on instances per model
MODEL_PORT_MAP = {}
port_counter = OLLAMA_START_PORT

for model in MODELS:
    instance_count = NUM_INSTANCES_PER_MODEL.get(model, 1)
    MODEL_PORT_MAP[model] = [port_counter + i for i in range(instance_count)]
    port_counter += instance_count

# ============================================
# Download Configuration
# ============================================
DEFAULT_MAX_DOWNLOAD_WORKERS = 5

# ============================================
# Debug & Logs
# ============================================
DEBUG_MODE = True

