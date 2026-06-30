"""Shared utilities for audio deepfake detection experiments."""

from pathlib import Path

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_metrics(metrics: list[dict], output_path: str | Path) -> None:
    """Save model evaluation metrics to a CSV file."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    pd.DataFrame(metrics).to_csv(output_path, index=False)

