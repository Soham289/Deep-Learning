"""BiLSTM Efficiency Experiment — Compare hidden sizes 128, 96, 64.

Uses shared pipeline, same seed, same splits, validation-based comparison only.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import json

from har_data import set_seed, load_bundle, make_loaders
from models.bilstm import BiLSTMClassifier
from train import train_model
from evaluate import evaluate_model, count_parameters


def run_config(hidden_size: int, seed: int = 42, verbose: bool = True):
    """Run a single BiLSTM configuration and return validation results."""
    set_seed(seed)

    DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
    bundle = load_bundle(DATA_ROOT, verbose=False)

    train_loader, val_loader, _ = make_loaders(
        bundle, batch_size=128, channels_first=False
    )

    model = BiLSTMClassifier(
        input_size=bundle["input_shape"][1],
        hidden_size=hidden_size,
        num_layers=2,
        n_classes=bundle["n_classes"],
        dropout=0.3,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Config: hidden_size={hidden_size}")
        print(f"Params: {count_parameters(model):,}")
        print(f"{'='*60}")

    result = train_model(
        model,
        train_loader,
        val_loader,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=100,
        patience=10,
        verbose=verbose,
    )

    # Evaluate on validation set for macro F1
    val_metrics = evaluate_model(model, val_loader, device=result["device"], class_names=bundle["class_names"])

    return {
        "hidden_size": hidden_size,
        "n_params": result["n_params"],
        "training_time_sec": result["training_time_sec"],
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "best_val_acc": result["history"]["val_acc"][result["best_epoch"] - 1],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_macro_precision": val_metrics["macro_precision"],
        "val_macro_recall": val_metrics["macro_recall"],
        "val_per_class_f1": val_metrics["per_class_f1"],
        "device": result["device"],
    }


def main():
    SEED = 42
    hidden_sizes = [128, 96, 64]

    all_results = []

    for hs in hidden_sizes:
        res = run_config(hs, seed=SEED, verbose=True)
        all_results.append(res)

    # Summary table
    print("\n\n" + "="*100)
    print("EFFICIENCY EXPERIMENT SUMMARY (VALIDATION METRICS)")
    print("="*100)
    print(f"{'Hidden':>6} | {'Params':>10} | {'Time(s)':>7} | {'Best Ep':>7} | {'Val Loss':>8} | {'Val Acc':>7} | {'Val F1':>7}")
    print("-"*100)
    for r in all_results:
        print(f"{r['hidden_size']:>6} | {r['n_params']:>10,} | {r['training_time_sec']:>7.1f} | {r['best_epoch']:>7} | "
              f"{r['best_val_loss']:>8.4f} | {r['best_val_acc']:>7.4f} | {r['val_macro_f1']:>7.4f}")

    # Per-class F1 on validation
    print("\n\nPer-class Validation F1:")
    print(f"{'Hidden':>6} | {'WALK':>6} | {'UP':>6} | {'DOWN':>6} | {'SIT':>6} | {'STAND':>6} | {'LAY':>6}")
    print("-"*70)
    for r in all_results:
        p = r["val_per_class_f1"]
        print(f"{r['hidden_size']:>6} | {p['WALKING']:>6.4f} | {p['WALKING_UPSTAIRS']:>6.4f} | "
              f"{p['WALKING_DOWNSTAIRS']:>6.4f} | {p['SITTING']:>6.4f} | {p['STANDING']:>6.4f} | {p['LAYING']:>6.4f}")

    # Save results
    out_path = PROJECT_ROOT / "results" / "metrics" / "bilstm_efficiency_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED,
            "val_subjects": [8, 16, 23, 29],
            "configurations": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Determine best config by validation loss (primary) then val macro F1 (secondary)
    best = min(all_results, key=lambda x: (x["best_val_loss"], -x["val_macro_f1"]))
    print(f"\n>> Best configuration by validation loss: hidden_size={best['hidden_size']}")
    print(f"   Val Loss: {best['best_val_loss']:.4f}, Val F1: {best['val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()