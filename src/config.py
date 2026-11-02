"""Resolve resource paths against the repository, independent of calling cwd."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or os.environ.get("SEM3D_CONFIG", ROOT / "config/resources.json"))
    # A local file may contain only paths; retain portable pipeline settings.
    config = json.loads((ROOT / "config/resources.example.json").read_text())
    for section, value in json.loads(path.read_text()).items():
        if isinstance(value, dict) and isinstance(config.get(section), dict):
            config[section].update(value)
        else:
            config[section] = value
    empty = [key for key, value in config["paths"].items()
             if not isinstance(value, str) or not value.strip()]
    if empty:
        raise ValueError(f"Fill these resource paths in {path}: " + ", ".join(f"paths.{key}" for key in empty))
    for key, value in config["paths"].items():
        value = os.path.expandvars(os.path.expanduser(value))
        config["paths"][key] = str((ROOT / value).absolute())
    return config


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
