import yaml
import os
from pathlib import Path


def _default_config() -> dict:
    # Minimal defaults required by backend request models and startup.
    return {
        "train": {
            "seed": 42,
        },
        "inference": {
            "default_mw": 5.5,
            "mw_min": 1.0,
            "mw_max": 10.0,
            "grid_size_x": 50,
            "grid_size_z": 50,
        },
    }

def _candidate_config_paths() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    env_path = os.getenv("SLIPGEN_CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Support both local workspace execution and the Docker mount layout.
    # The first matching config.yaml wins.
    for base_dir in [script_dir, *script_dir.parents]:
        candidates.append(base_dir / "config.yaml")

    return candidates


def _resolve_config_path() -> Path:
    for candidate in _candidate_config_paths():
        if candidate.is_file():
            return candidate.resolve()
    return Path("")


CONFIG_PATH = _resolve_config_path()

def load_config():
    if CONFIG_PATH and CONFIG_PATH.is_file():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    checked_paths = ", ".join(str(path) for path in _candidate_config_paths())
    print(f"Warning: Configuration file not found. Checked: {checked_paths}. Using built-in defaults.")
    return _default_config()

config = load_config()
