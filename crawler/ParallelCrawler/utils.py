# utils.py

import os
import json
import itertools

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

def load_processed_dataset_ids(filepath):
    ids = load_json(filepath, default=[])
    return set(ids)

def save_processed_dataset_ids(filepath, ids):
    save_json(filepath, list(ids))

# Model Port Cycling Helper
def get_port_cycle(ports):
    """Creates an infinite cycle iterator over model ports."""
    return itertools.cycle(ports)

