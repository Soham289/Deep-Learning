"""Shared evaluation entry point for metrics and confusion matrices.

Every model reports through this one function so the comparison table in the
report (accuracy, macro P/R/F1, per-class F1, confusion matrix) is computed
identically for all three architectures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def evaluate_model(model, loader, device: str | torch.device | None, class_names) -> dict:
    """Run inference over `loader` and compute the shared metric set.

    Returns accuracy, macro-averaged precision/recall/F1, per-class F1, the
    raw confusion matrix, and the full sklearn classification report string
    (kept for the appendix / notebook output).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    all_preds, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class_f1": {name: report[name]["f1-score"] for name in class_names},
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }


def count_parameters(model) -> int:
    """Trainable parameter count, for the comparison table's model-size column."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_confusion_matrix(cm, class_names, save_path: str | Path, title: str = "Confusion matrix") -> None:
    """Save a normalised confusion-matrix heatmap to `save_path`.

    Normalised by true-class row so classes with fewer test windows aren't
    visually flattened by classes with more — important here since the sitting
    vs. standing confusion (the failure mode this project is meant to explain)
    is otherwise easy to miss next to the much larger walking classes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=np.float64)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Import evaluate_model from this module; it is not a standalone script.")
