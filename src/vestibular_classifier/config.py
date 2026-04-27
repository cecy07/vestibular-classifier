"""YAML config loaders for coefficients and post-processing settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def _load_yaml(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_coefficients(path: str | Path | None = None) -> dict[str, float]:
    """Load the tuned rule coefficients from YAML.

    Defaults to ``configs/coefficients.yaml`` at the repository root.
    """
    path = Path(path) if path is not None else _CONFIG_DIR / "coefficients.yaml"
    data = _load_yaml(path)
    # Ensure all values are numeric so downstream math behaves.
    return {k: float(v) for k, v in data.items()}


def load_postprocessing(path: str | Path | None = None) -> dict[str, Any]:
    """Load the post-processing switches and vascular-screen parameters."""
    path = Path(path) if path is not None else _CONFIG_DIR / "postprocessing.yaml"
    return _load_yaml(path)
