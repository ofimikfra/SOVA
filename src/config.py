"""
SOVA Config
Reads and writes config.json in the project root.
All other modules import settings from here.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

DEFAULTS = {
    "ollama_model":   "llama3.2:3b",
    "flush_interval": 30,
    "tts_enabled":    True,
    "tts_volume":     0.5,      # 0.0–1.0
}


def load() -> dict:
    if not os.path.exists(CONFIG_PATH):
        save(DEFAULTS)
        return DEFAULTS.copy()
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        for k, v in DEFAULTS.items():
            data.setdefault(k, v)
        return data
    except (json.JSONDecodeError, IOError):
        return DEFAULTS.copy()


def save(config: dict):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        print(f"[CONFIG] Failed to save: {e}")


def update(partial: dict):
    """Merge partial dict into existing config and save."""
    current = load()
    current.update(partial)
    save(current)
    return current