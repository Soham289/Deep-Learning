"""Leave-one-subject-out (LOSO) cross-validation over all 30 volunteers.

Each fold holds one volunteer out as the test set, picks 4 validation subjects
at random (deterministically, per fold) from the remaining 29 for early
stopping, trains on the other 25, and fits the standardiser on those 25 only.
No test or validation statistic reaches training.

Two things come out of it:
  * a score that does not hinge on one particular 9-subject test set, with
    per-subject variance;
  * per-subject accuracy, which shows whether the fixed validation subjects
    (8, 16, 23, 29) are harder than average -- the suspected cause of the
    early-stopping epoch and the validation/test accuracy gap.

    python src/run_loso.py --model cnn1d
    python src/run_loso.py --model cnn1d --subjects 1 2 --max-epochs 2   # smoke test

Progress is saved after every fold, so an interrupted run resumes where it
stopped. Writes results/metrics/<model>_loso.json and
results/figures/<model>_loso_per_subject.png.

Note: the MLP uses BatchNorm1d on flattened input, which raises if a final
training batch has exactly one sample, and LOSO fold sizes can produce one. The
training loader in har_data.make_loaders now drops a singleton trailing batch,
so LOSO runs for every model.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from experiments import RAW_MODELS, SPECS, run_once  # noqa: E402
from har_data import (ACTIVITY_NAMES, DEFAULT_VAL_SUBJECTS, N_CLASSES,  # noqa: E402
                      WINDOW_LEN, N_CHANNELS, ChannelStandardizer, load_all,
                      load_subjects)

DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
SEED = 42
N_VAL = 4


def fold_bundle(X, y, s, held_out: int) -> dict:
    """Build a har_data-style bundle for one LOSO fold (scaler fit on train only)."""
    others = np.array(sorted(set(s.tolist()) - {held_out}))
    rng = np.random.default_rng(1000 + held_out)
    val_subj = tuple(sorted(rng.choice(others, size=N_VAL, replace=False).tolist()))

    te = s == held_out
    va = np.isin(s, list(val_subj))
    tr = ~te & ~va

    scaler = ChannelStandardizer()
    X_tr = scaler.fit_transform(X[tr])
    return {
        "X_train": X_tr, "y_train": y[tr],
        "X_val": scaler.transform(X[va]), "y_val": y[va],
        "X_test": scaler.transform(X[te]), "y_test": y[te],
        "class_names": ACTIVITY_NAMES,
        "input_shape": (WINDOW_LEN, N_CHANNELS),
        "n_classes": N_CLASSES,
        "val_subjects": val_subj,
        "n_train_subjects": len(np.unique(s[tr])),
    }


def summarize(name: str, folds: dict) -> dict:
    subs = sorted(int(k) for k in folds)
    acc = np.array([folds[str(k)]["test_accuracy"] for k in subs])
    cm = sum(np.asarray(folds[str(k)]["confusion_matrix"]) for k in subs)
    cm = np.asarray(cm, dtype=float)

    recall = np.diag(cm) / cm.sum(axis=1)
    precision = np.diag(cm) / cm.sum(axis=0)
    f1 = 2 * precision * recall / (precision + recall)

    train_subj = set(load_subjects(DATA_ROOT, "train").tolist())
    test_subj = set(load_subjects(DATA_ROOT, "test").tolist())
    groups = {
        "official_val_subjects": [k for k in subs if k in DEFAULT_VAL_SUBJECTS],
        "other_official_train_subjects": [k for k in subs if k in train_subj - set(DEFAULT_VAL_SUBJECTS)],
        "official_test_subjects": [k for k in subs if k in test_subj],
    }
    by_group = {
        g: {"subjects": ks,
            "mean_accuracy": float(np.mean([folds[str(k)]["test_accuracy"] for k in ks])) if ks else None}
        for g, ks in groups.items()
    }
    return {
        "model": SPECS[name]["label"],
        "n_folds": len(subs),
        "per_subject_accuracy": {str(k): float(a) for k, a in zip(subs, acc)},
        "accuracy_mean_over_subjects": float(acc.mean()),
        "accuracy_std_over_subjects": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "accuracy_min": {"subject": subs[int(acc.argmin())], "value": float(acc.min())},
        "accuracy_max": {"subject": subs[int(acc.argmax())], "value": float(acc.max())},
        "pooled_accuracy": float(np.trace(cm) / cm.sum()),
        "pooled_macro_f1": float(f1.mean()),
        "pooled_per_class_f1": {c: float(v) for c, v in zip(ACTIVITY_NAMES, f1)},
        "pooled_sitting_as_standing": float(cm[3, 4] / cm[3].sum()),
        "pooled_standing_as_sitting": float(cm[4, 3] / cm[4].sum()),
        "pooled_confusion_matrix": cm.astype(int).tolist(),
        "best_epoch_mean": float(np.mean([folds[str(k)]["best_epoch"] for k in subs])),
        "training_time_sec_mean": float(np.mean([folds[str(k)]["training_time_sec"] for k in subs])),
        "group_accuracy": by_group,
    }


def plot(name: str, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = {int(k): v for k, v in summary["per_subject_accuracy"].items()}
    g = summary["group_accuracy"]
    color = {}
    for k in acc:
        color[k] = ("tab:red" if k in g["official_val_subjects"]["subjects"]
                    else "tab:green" if k in g["official_test_subjects"]["subjects"]
                    else "tab:blue")
    subs = sorted(acc)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([str(k) for k in subs], [acc[k] for k in subs], color=[color[k] for k in subs])
    ax.axhline(summary["accuracy_mean_over_subjects"], color="black", linestyle="--", linewidth=1)
    ax.set_ylim(min(acc.values()) - 0.05, 1.0)
    ax.set_xlabel("held-out subject")
    ax.set_ylabel("accuracy on that subject")
    ax.set_title(f"{summary['model']} LOSO per-subject accuracy "
                 "(red = fixed validation subjects, green = official test subjects)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}_loso_per_subject.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # Folds are rebuilt from the raw inertial signals, so only the raw-input
    # models can be run here; the 561-feature MLP has no per-fold feature split.
    ap.add_argument("--model", choices=list(RAW_MODELS), required=True)
    ap.add_argument("--subjects", type=int, nargs="+", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    args = ap.parse_args()

    X, y, s = load_all(DATA_ROOT)
    todo = args.subjects or sorted(np.unique(s).tolist())
    out = args.out_dir / f"{args.model}_loso.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = json.loads(out.read_text()) if out.exists() else {"folds": {}}
    folds = state["folds"]
    train_kwargs = {"max_epochs": args.max_epochs} if args.max_epochs else None

    for subj in todo:
        if str(subj) in folds:
            print(f"subject {subj}: already done, skipping")
            continue
        b = fold_bundle(X, y, s, subj)
        r = run_once(args.model, b, SEED, train_kwargs=train_kwargs)
        r["subject"] = subj
        r["val_subjects"] = list(b["val_subjects"])
        r["n_test_windows"] = int(len(b["y_test"]))
        folds[str(subj)] = r
        out.write_text(json.dumps({"model": SPECS[args.model]["label"], "folds": folds}, indent=2))
        print(f"subject {subj:2d}: acc {r['test_accuracy']:.4f}  best epoch {r['best_epoch']:2d}  "
              f"{r['training_time_sec']:.0f}s  ({len(folds)}/{len(todo)})", flush=True)

    summary = summarize(args.model, folds)
    out.write_text(json.dumps({"model": SPECS[args.model]["label"], "summary": summary,
                               "folds": folds}, indent=2))
    plot(args.model, summary)
    print(f"\nLOSO ({summary['n_folds']} folds): per-subject acc "
          f"{summary['accuracy_mean_over_subjects']:.4f} +/- {summary['accuracy_std_over_subjects']:.4f}, "
          f"pooled acc {summary['pooled_accuracy']:.4f}, pooled macro-F1 {summary['pooled_macro_f1']:.4f}")
    for g, v in summary["group_accuracy"].items():
        print(f"  {g:32s} mean acc {v['mean_accuracy']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
