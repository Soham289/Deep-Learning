"""Final BiLSTM Evaluation — hidden_size=96 (selected by validation loss)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import json
import matplotlib.pyplot as plt

import torch

from har_data import set_seed, load_bundle, make_loaders
from models.bilstm import BiLSTMClassifier
from train import train_model
from evaluate import evaluate_model, count_parameters, plot_confusion_matrix


def main():
    SEED = 42
    HIDDEN_SIZE = 96

    set_seed(SEED)

    DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
    RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
    RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(DATA_ROOT)
    class_names = bundle["class_names"]

    train_loader, val_loader, test_loader = make_loaders(
        bundle, batch_size=128, channels_first=False
    )

    model = BiLSTMClassifier(
        input_size=bundle["input_shape"][1],
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        n_classes=bundle["n_classes"],
        dropout=0.3,
    )

    print(f"\n{'='*60}")
    print(f"Final BiLSTM — hidden_size={HIDDEN_SIZE}")
    print(f"Params: {count_parameters(model):,}")
    print(f"{'='*60}\n")

    # Train
    result = train_model(
        model,
        train_loader,
        val_loader,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=100,
        patience=10,
    )

    print(f"\nbest epoch: {result['best_epoch']}")
    print(f"best val loss: {result['best_val_loss']:.4f}")
    print(f"training time: {result['training_time_sec']:.1f}s")
    print(f"trainable parameters: {result['n_params']:,}")
    print(f"device: {result['device']}")

    # Training curves
    history = result["history"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].axvline(result["best_epoch"] - 1, color="grey", linestyle="--", label="best epoch")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(history["val_acc"], color="tab:green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy"); ax[1].set_title("Validation accuracy")
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "bilstm_96_training_curves.png", dpi=150)
    plt.close(fig)

    # Test evaluation
    metrics = evaluate_model(model, test_loader, device=result["device"], class_names=class_names)

    print(f"\nTest Results:")
    print(f"  accuracy       : {metrics['accuracy']:.4f}")
    print(f"  macro precision: {metrics['macro_precision']:.4f}")
    print(f"  macro recall   : {metrics['macro_recall']:.4f}")
    print(f"  macro F1       : {metrics['macro_f1']:.4f}")
    print()
    print(metrics["classification_report"])

    # Confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        save_path=RESULTS_FIGURES / "bilstm_96_confusion_matrix.png",
        title="BiLSTM (hidden=96) — normalised confusion matrix (test set)",
    )
    print(f"\nConfusion matrix saved to {RESULTS_FIGURES / 'bilstm_96_confusion_matrix.png'}")

    # Save final metrics
    summary = {
        "model": "BiLSTM",
        "owner": "Avni",
        "hidden_size": HIDDEN_SIZE,
        "seed": SEED,
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

    out_path = RESULTS_METRICS / "bilstm_96_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    return summary


if __name__ == "__main__":
    main()