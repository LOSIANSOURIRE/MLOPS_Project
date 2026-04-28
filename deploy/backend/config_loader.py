import yaml
import os
from pathlib import Path

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
    raise FileNotFoundError(
        "Configuration file not found. Checked: "
        + ", ".join(str(path) for path in _candidate_config_paths())
    )


CONFIG_PATH = _resolve_config_path()

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
