"""Shared helpers for multi-run experiments (seeds, LOSO, ablations).

Holds each raw-signal model's construction and the training configuration its
owner settled on, so every experiment runner trains the three architectures
exactly as their notebooks/drivers do. Nothing here changes the shared protocol:
data comes from `har_data.load_bundle`, training from `train.train_model`,
scoring from `evaluate.evaluate_model`.

Only the raw (N, 128, 9) models are registered. The 561-feature MLP uses its own
loaders (see notebooks/member_a_mlp_driver.py) and is a separate representation
study, not a raw-input architecture result.
"""

from __future__ import annotations

import numpy as np

from evaluate import evaluate_model
from har_data import make_loaders, set_seed
from models.bilstm import BiLSTMClassifier
from models.cnn1d import HARCnn1D
from models.mlp import HARMlp
from train import train_model

SITTING, STANDING = 3, 4  # indices in har_data.ACTIVITY_NAMES

SPECS = {
    "cnn1d": {
        "label": "1D-CNN",
        "channels_first": True,
        "build": lambda b, **kw: HARCnn1D(
            in_channels=b["input_shape"][1], n_classes=b["n_classes"], **kw),
        "train": dict(lr=3e-4, weight_decay=1e-4, max_epochs=100, patience=15),
    },
    "mlp": {
        "label": "MLP (raw signals)",
        "channels_first": False,
        "build": lambda b, **kw: HARMlp(
            input_dim=b["input_shape"][0] * b["input_shape"][1],
            n_classes=b["n_classes"], **kw),
        "train": dict(lr=1e-3, weight_decay=1e-4, max_epochs=100, patience=15),
    },
    "bilstm": {
        "label": "BiLSTM (hidden 96)",
        "channels_first": False,
        "build": lambda b, **kw: BiLSTMClassifier(
            **{"input_size": b["input_shape"][1], "hidden_size": 96,
               "num_layers": 2, "n_classes": b["n_classes"], "dropout": 0.3, **kw}),
        "train": dict(lr=1e-3, weight_decay=0.0, max_epochs=100, patience=10),
    },
}


def run_once(name: str, bundle: dict, seed: int, model_kwargs: dict | None = None,
             train_kwargs: dict | None = None, batch_size: int = 128,
             verbose: bool = False) -> dict:
    """Train one model from scratch at `seed` and score it on the bundle's test set."""
    spec = SPECS[name]
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(
        bundle, batch_size=batch_size, channels_first=spec["channels_first"])
    model = spec["build"](bundle, **(model_kwargs or {}))
    cfg = {**spec["train"], **(train_kwargs or {})}
    res = train_model(model, train_loader, val_loader, verbose=verbose, **cfg)
    m = evaluate_model(model, test_loader, device=res["device"],
                       class_names=bundle["class_names"])

    cm = np.asarray(m["confusion_matrix"], dtype=float)
    hist = res["history"]
    return {
        "seed": seed,
        "n_params": res["n_params"],
        "training_time_sec": res["training_time_sec"],
        "best_epoch": res["best_epoch"],
        "epochs_run": len(hist["val_loss"]),
        "best_val_loss": res["best_val_loss"],
        "best_val_acc": hist["val_acc"][res["best_epoch"] - 1],
        "test_accuracy": m["accuracy"],
        "test_macro_precision": m["macro_precision"],
        "test_macro_recall": m["macro_recall"],
        "test_macro_f1": m["macro_f1"],
        "per_class_f1": m["per_class_f1"],
        "sitting_as_standing": cm[SITTING, STANDING] / cm[SITTING].sum(),
        "standing_as_sitting": cm[STANDING, SITTING] / cm[STANDING].sum(),
        "confusion_matrix": m["confusion_matrix"],
    }


SCALAR_KEYS = (
    "n_params", "training_time_sec", "best_epoch", "epochs_run", "best_val_loss",
    "best_val_acc", "test_accuracy", "test_macro_precision", "test_macro_recall",
    "test_macro_f1", "sitting_as_standing", "standing_as_sitting",
)


def aggregate(records: list[dict]) -> dict:
    """Mean / sample std (ddof=1) / min / max across runs, incl. per-class F1."""
    def stats(vals):
        v = np.asarray(vals, dtype=float)
        return {"mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}

    out = {k: stats([r[k] for r in records]) for k in SCALAR_KEYS}
    classes = records[0]["per_class_f1"].keys()
    out["per_class_f1"] = {c: stats([r["per_class_f1"][c] for r in records]) for c in classes}
    out["n_runs"] = len(records)
    return out
