"""Member A driver — trains the MLP baseline on both input representations.

This is plumbing, deliberately mirroring Bhavya's member_b_cnn1d.ipynb so the
two runs are directly comparable. The thinking you are graded on lives in
src/models/mlp.py (the architecture) and in how you read the results at the
bottom — not here.

Run from the repo root once src/models/mlp.py is implemented:

    python notebooks/member_a_mlp_driver.py

Writes results/metrics/mlp_metrics.json, mlp561_metrics.json and the two
confusion matrices + training curves into results/figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")  # headless-safe; drop this line if running in a notebook
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from har_data import set_seed, load_bundle
from models.mlp import HARMlp
from train import train_model
from evaluate import evaluate_model, count_parameters, plot_confusion_matrix

DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128


def feature_loaders(bundle: dict, batch_size: int = BATCH_SIZE):
    """DataLoaders over the 561 hand-crafted features.

    har_data.make_loaders only handles the (N, 128, 9) windows, so the feature
    path needs its own loaders. Same subjects, same split, same seed — only the
    representation changes, which is the whole point of the comparison.

    The 561 features ship pre-normalised to [-1, 1], so they are used as-is.
    Note this in the report: it is a deliberate choice, not an oversight.
    """
    def ds(F, y):
        return TensorDataset(torch.from_numpy(np.ascontiguousarray(F)), torch.from_numpy(y))

    common = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    return (
        DataLoader(ds(bundle["F_train"], bundle["y_train"]), shuffle=True, **common),
        DataLoader(ds(bundle["F_val"], bundle["y_val"]), shuffle=False, **common),
        DataLoader(ds(bundle["F_test"], bundle["y_test"]), shuffle=False, **common),
    )


def window_loaders(bundle: dict, batch_size: int = BATCH_SIZE):
    """DataLoaders over raw (N, 128, 9) windows — channels_first=False."""
    def ds(X, y):
        return TensorDataset(torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(y))

    common = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    return (
        DataLoader(ds(bundle["X_train"], bundle["y_train"]), shuffle=True, **common),
        DataLoader(ds(bundle["X_val"], bundle["y_val"]), shuffle=False, **common),
        DataLoader(ds(bundle["X_test"], bundle["y_test"]), shuffle=False, **common),
    )


def plot_curves(result: dict, tag: str, title: str) -> None:
    h = result["history"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["train_loss"], label="train")
    ax[0].plot(h["val_loss"], label="val")
    ax[0].axvline(result["best_epoch"] - 1, color="grey", linestyle="--", label="best epoch")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(h["val_acc"], color="tab:green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy"); ax[1].set_title("Validation accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{tag}_training_curves.png", dpi=150)
    plt.close(fig)


def run(name: str, tag: str, input_dim: int, loaders, bundle: dict, lr: float) -> dict:
    """Train, evaluate and persist one variant."""
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    set_seed(SEED)  # re-seeded per variant so the two runs are independent

    train_loader, val_loader, test_loader = loaders
    model = HARMlp(input_dim=input_dim, n_classes=bundle["n_classes"])
    print(f"trainable parameters: {count_parameters(model):,}\n")

    result = train_model(
        model, train_loader, val_loader,
        lr=lr, weight_decay=1e-4, max_epochs=100, patience=15,
    )
    print(f"\nbest epoch {result['best_epoch']} | "
          f"val loss {result['best_val_loss']:.4f} | "
          f"{result['training_time_sec']:.1f}s on {result['device']}")

    plot_curves(result, tag, f"{name} — training curves")

    metrics = evaluate_model(model, test_loader, device=result["device"],
                             class_names=bundle["class_names"])
    print(f"\ntest accuracy : {metrics['accuracy']:.4f}")
    print(f"macro F1      : {metrics['macro_f1']:.4f}")
    print()
    print(metrics["classification_report"])

    plot_confusion_matrix(
        metrics["confusion_matrix"], bundle["class_names"],
        save_path=FIGURES_DIR / f"{tag}_confusion_matrix.png",
        title=f"{name} — normalised confusion matrix (test set)",
    )

    payload = {
        "model": name,
        "owner": "Soham",
        "seed": SEED,
        "input_representation": "raw 128x9 window" if input_dim != 561 else "561 hand-crafted features",
        "val_subjects": list(bundle["val_subjects"]),
        "n_params": result["n_params"],
        "training_time_sec": result["training_time_sec"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        "test_macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    out = METRICS_DIR / f"{tag}_metrics.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out.relative_to(PROJECT_ROOT)}")
    return payload


def main() -> None:
    set_seed(SEED)
    bundle = load_bundle(DATA_ROOT, with_features561=True)

    raw = run("MLP (raw signals)", "mlp", 128 * 9,
              window_loaders(bundle), bundle, lr=1e-3)

    feat = run("MLP (561 features)", "mlp561", 561,
               feature_loaders(bundle), bundle, lr=1e-3)

    print(f"\n{'=' * 70}\nDual-representation summary\n{'=' * 70}")
    print(f"  raw 128x9 signals   : {raw['test_accuracy']:.4f} acc | "
          f"macro F1 {raw['test_macro_f1']:.4f}")
    print(f"  561 hand-crafted    : {feat['test_accuracy']:.4f} acc | "
          f"macro F1 {feat['test_macro_f1']:.4f}")
    gap = feat["test_accuracy"] - raw["test_accuracy"]
    print(f"  gap                 : {gap:+.4f}")
    print("\n  Same architecture, same subjects, same seed. The gap is the value")
    print("  of the hand-engineered representation -- and the CNN's advantage")
    print("  over the raw baseline is it learning features of that kind itself.")


if __name__ == "__main__":
    main()
