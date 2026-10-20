"""Is the fixed 4-subject validation set noisy? (1D-CNN, Member B)

The interim report flagged that every model reached its best validation loss at
epoch 5-6 and asked whether that reflects a too-high learning rate or a small,
noisy validation set. The CNN's validation accuracy hovers around 85% while its
test accuracy is about 93%, which points at the validation set.

This trains the CNN once per alternative validation split -- the 21 training
subjects partitioned into five disjoint subject groups (5, 4, 4, 4, 4), each
taking a turn as validation -- plus the official split (8, 16, 23, 29). Same
seed, same hyperparameters, same test set. If validation accuracy and best
epoch swing widely between splits while test accuracy stays put, the validation
set (not the optimiser) is the unstable part.

    python notebooks/cnn1d_validation_noise_study.py

Writes results/metrics/cnn1d_validation_noise.json and
results/figures/cnn1d_validation_noise.png.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments import run_once  # noqa: E402
from har_data import DEFAULT_VAL_SUBJECTS, load_bundle, load_subjects  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
SEED = 42


def make_splits() -> list[tuple[str, tuple[int, ...]]]:
    train_subj = np.array(sorted(set(load_subjects(DATA_ROOT, "train").tolist())))
    perm = np.random.default_rng(0).permutation(train_subj)
    sizes = [5, 4, 4, 4, 4]
    splits, start = [], 0
    for i, n in enumerate(sizes, 1):
        splits.append((f"group {i}", tuple(sorted(perm[start:start + n].tolist()))))
        start += n
    splits.append(("official", tuple(DEFAULT_VAL_SUBJECTS)))
    return splits


def main() -> None:
    rows = []
    for label, val_subj in make_splits():
        bundle = load_bundle(DATA_ROOT, val_subjects=val_subj, verbose=False)
        r = run_once("cnn1d", bundle, SEED)
        rows.append({
            "split": label, "val_subjects": list(val_subj),
            "n_val_windows": int(len(bundle["y_val"])),
            "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
            "best_val_loss": r["best_val_loss"], "best_val_acc": r["best_val_acc"],
            "test_accuracy": r["test_accuracy"], "test_macro_f1": r["test_macro_f1"],
        })
        print(f"{label:9s} val={list(val_subj)}  best epoch {r['best_epoch']:2d}  "
              f"val acc {r['best_val_acc']:.4f}  val loss {r['best_val_loss']:.4f}  "
              f"test acc {r['test_accuracy']:.4f}", flush=True)

    def spread(k):
        v = np.array([x[k] for x in rows])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "min": float(v.min()), "max": float(v.max())}

    summary = {k: spread(k) for k in ("best_epoch", "best_val_loss", "best_val_acc", "test_accuracy")}
    summary["mean_val_minus_test_accuracy"] = float(
        np.mean([x["best_val_acc"] - x["test_accuracy"] for x in rows]))
    out = METRICS_DIR / "cnn1d_validation_noise.json"
    out.write_text(json.dumps({"seed": SEED, "splits": rows, "summary": summary}, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [r["best_val_acc"] for r in rows], 0.4, label="validation")
    ax[0].bar(x + 0.2, [r["test_accuracy"] for r in rows], 0.4, label="test")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([r["split"] for r in rows], rotation=20)
    ax[0].set_ylim(0.75, 1.0)
    ax[0].set_ylabel("accuracy at best epoch")
    ax[0].legend()
    ax[0].set_title("Validation vs test accuracy by validation split")
    ax[1].bar(x, [r["best_epoch"] for r in rows], color="tab:purple")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([r["split"] for r in rows], rotation=20)
    ax[1].set_ylabel("best epoch")
    ax[1].set_title("Early-stopping epoch by validation split")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cnn1d_validation_noise.png", dpi=150)
    plt.close(fig)

    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
