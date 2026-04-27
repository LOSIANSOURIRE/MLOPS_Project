import yaml
import os
import sys
from pathlib import Path

# Dynamically resolve root project path relative to this script
# deploy/backend/config_loader.py -> ../../../config.yaml
DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
)

CONFIG_PATH = os.path.abspath(
    os.getenv("SLIPGEN_CONFIG_PATH", DEFAULT_CONFIG_PATH)
)

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()
