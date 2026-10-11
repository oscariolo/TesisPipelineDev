"""Helpers for turning model identifiers into safe, predictable filesystem names."""

from __future__ import annotations

import re
from pathlib import Path

_UNSAFE_PATH_CHARS = str.maketrans({"/": "-", "\\": "-", ":": "-", " ": "_", "|": "-"})


def sanitize_model_name(model_name: str) -> str:
    """Turn a model/repo id into a single safe directory name ('org/model' -> 'org-model')."""
    cleaned = str(model_name).strip().translate(_UNSAFE_PATH_CHARS)
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    return cleaned or "unknown-model"


def model_output_dir(analysis_dir: str | Path, model_name: str | None) -> Path:
    """Folder holding one model's pipeline output and its evaluation.

    Always ``<analysis_dir>/<sanitized model_name>``; without a model name the analysis dir itself
    is used, so a run that cannot be attributed to a model still writes somewhere predictable.
    """
    base = Path(analysis_dir)
    if not model_name:
        return base
    return base / sanitize_model_name(model_name)
