"""Evaluation and plotting helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from utils import ensure_directory


def plot_confusion_matrix(y_true, y_pred, output_path: str | Path) -> None:
    """Save a confusion matrix plot."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=["Real", "Fake"])
    display.plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_comparison(metrics, output_path: str | Path) -> None:
    """Save a bar chart comparing model F1 scores."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    names = [row["model"] for row in metrics]
    f1_scores = [row["f1"] for row in metrics]

    plt.figure(figsize=(8, 5))
    plt.bar(names, f1_scores, color="#4C78A8")
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("Model Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

