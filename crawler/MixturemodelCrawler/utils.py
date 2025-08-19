# utils.py

import os
import json
import itertools
import random
from config import DEBUG_MODE, MODEL_PORT_MAP

# ============================================
# JSON Load/Save Helpers
# ============================================

def ensure_folder_exists(folder):
    os.makedirs(folder, exist_ok=True)

def load_json(filepath, default=None):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return default or {}

def save_json(filepath, data):
    ensure_folder_exists(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# ============================================
# Processed Dataset Tracking Helpers
# ============================================

def load_processed_dataset_ids(filepath):
    ids = load_json(filepath, default=[])
    return set(ids)

def save_processed_dataset_ids(filepath, ids):
    save_json(filepath, list(ids))

# ============================================
# Model and Port Cycling Helpers
# ============================================

def get_model_cycle(models):
    """
    Create an infinite cycle of the selected models.
    """
    return itertools.cycle(models)

def get_port_cycle_for_model(model_name):
    """
    Create an infinite cycle of ports for a given model.
    """
    ports = MODEL_PORT_MAP.get(model_name, [])
    if not ports:
        raise ValueError(f"No ports found for model {model_name}. Check MODEL_PORT_MAP.")
    return itertools.cycle(ports)

# ============================================
# Random Model Selection for Each Run
# ============================================

def select_random_models(all_models, num_models_to_use):
    """
    Select a random subset of models from the available model list.
    """
    if num_models_to_use > len(all_models):
        raise ValueError("NUM_MODELS_TO_USE exceeds number of available models.")
    
    selected_models = random.sample(all_models, num_models_to_use)
    
    print(f"🎲 Selected models for this run: {selected_models}")
    return selected_models

# ============================================
# Debug Logging
# ============================================

def debug_log(message):
    """
    Prints debug messages only if DEBUG_MODE is enabled.
    """
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

