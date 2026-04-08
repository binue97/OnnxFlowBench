"""
Project configuration loader.

Reads ``datasets.yaml`` from the project root to resolve dataset paths.
Falls back to hardcoded defaults when the file is missing or a key is absent.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

_CONFIG_PATH = Path(__file__).resolve().parent / "datasets.yaml"


@lru_cache(maxsize=1)
def _load_datasets() -> dict[str, str]:
    """Load the datasets section from datasets.yaml (cached)."""
    if not _CONFIG_PATH.is_file():
        return {}

    import yaml

    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return {}
    datasets = data.get("datasets", {})
    return datasets if isinstance(datasets, dict) else {}


def get_dataset_root(name: str, fallback: str) -> str:
    """Return the configured root for *name*, or *fallback* if unset."""
    return _load_datasets().get(name, fallback)
