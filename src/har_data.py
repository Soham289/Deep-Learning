"""
har_data.py — shared data pipeline for the ICT 4442 HAR mini project.

Every model (MLP / 1D-CNN / BiLSTM) imports from this module so that all three
are trained and evaluated on byte-identical data. Do not fork this file.

Protocol enforced here
----------------------
* Windows are 128 timesteps x 9 channels, taken from the raw Inertial Signals
  files (NOT the 561 pre-computed features).
* The official UCI train/test partition is subject-wise: the 30 volunteers are
  split 21 / 9, and no subject appears in both. `verify_subject_independence()`
  checks this against the actual files rather than trusting the README.
* Validation subjects are held out from the 21 TRAINING subjects only.
* Standardisation statistics are fitted on the training subjects alone and then
  applied unchanged to validation and test. Fitting on anything wider leaks.

Usage
-----
    from har_data import load_bundle, set_seed

    set_seed(42)
    d = load_bundle("data/UCI HAR Dataset")
    d["X_train"].shape        # (N, 128, 9), standardised
    d["y_train"].shape        # (N,), int labels 0..5

Run `python har_data.py --root "data/UCI HAR Dataset"` for a sanity report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Channel order. Every model sees the 9 channels in exactly this sequence.
CHANNELS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)

#: Raw labels in the y_*.txt files are 1..6; we shift to 0..5 for PyTorch.
ACTIVITY_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)

WINDOW_LEN = 128
N_CHANNELS = len(CHANNELS)
N_CLASSES = len(ACTIVITY_NAMES)

#: Held out from the 21 training subjects. Fixed so all three members get the
#: same validation set. Cite these IDs in the Experimental Setup section.
DEFAULT_VAL_SUBJECTS = (8, 16, 23, 29)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG in play and put cuDNN into deterministic mode.

    The whole project rests on the claim that architecture is the only variable
    that differs between models, so non-determinism undermines the argument.
    Call this at the top of every training script.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def _resolve_root(root: str | Path) -> Path:
    """Accept either the dataset dir or its parent, and fail loudly if wrong."""
    root = Path(root)
    if not (root / "train").is_dir() and (root / "UCI HAR Dataset").is_dir():
        root = root / "UCI HAR Dataset"
    missing = [p for p in ("train", "test", "activity_labels.txt")
               if not (root / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{root} does not look like the UCI HAR Dataset folder "
            f"(missing: {', '.join(missing)}). Download the full zip from "
            "https://archive.ics.uci.edu/dataset/240/ — the Kaggle CSV mirror "
            "does not contain the Inertial Signals the CNN and LSTM need."
        )
    return root


def load_signals(root: str | Path, split: str) -> np.ndarray:
    """Stack the nine Inertial Signals files into (N, 128, 9) float32."""
    root = _resolve_root(root)
    sig_dir = root / split / "Inertial Signals"
    if not sig_dir.is_dir():
        raise FileNotFoundError(
            f"{sig_dir} not found. This is the folder the Kaggle mirror omits."
        )
    cols = []
    for ch in CHANNELS:
        arr = np.loadtxt(sig_dir / f"{ch}_{split}.txt", dtype=np.float32)
        if arr.shape[1] != WINDOW_LEN:
            raise ValueError(f"{ch}_{split}.txt has {arr.shape[1]} cols, expected {WINDOW_LEN}")
        cols.append(arr)
    return np.stack(cols, axis=-1)  # (N, 128, 9)


def load_labels(root: str | Path, split: str) -> np.ndarray:
    """Load y_*.txt and shift from 1..6 to 0..5."""
    root = _resolve_root(root)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64)
    return y - 1


def load_subjects(root: str | Path, split: str) -> np.ndarray:
    """Load the volunteer ID for each window."""
    root = _resolve_root(root)
    return np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)


def load_features561(root: str | Path, split: str) -> np.ndarray:
    """Load the 561 hand-crafted features (Member A's secondary comparison).

    Not used by the CNN or LSTM — they consume the raw signals.
    """
    root = _resolve_root(root)
    return np.loadtxt(root / split / f"X_{split}.txt", dtype=np.float32)


# ---------------------------------------------------------------------------
# The claim that has to be checked, not assumed
# ---------------------------------------------------------------------------

def verify_subject_independence(root: str | Path, verbose: bool = True) -> dict:
    """Confirm from the files that no volunteer appears in both partitions.

    The ESANN paper's wording is ambiguous about whether the 70/30 split is over
    windows or over volunteers. The distinction matters enormously: because
    consecutive windows overlap by 50%, a window-level split would place
    near-duplicates on both sides and inflate every number we report. This
    function settles it empirically. Put the output in the report.
    """
    tr = set(load_subjects(root, "train").tolist())
    te = set(load_subjects(root, "test").tolist())
    overlap = tr & te
    result = {
        "n_train_subjects": len(tr),
        "n_test_subjects": len(te),
        "overlap": sorted(overlap),
        "subject_independent": len(overlap) == 0,
        "train_subjects": sorted(tr),
        "test_subjects": sorted(te),
    }
    if verbose:
        print(f"  train subjects : {len(tr)}  {sorted(tr)}")
        print(f"  test  subjects : {len(te)}  {sorted(te)}")
        if overlap:
            print(f"  !! OVERLAP     : {sorted(overlap)} — split is NOT subject-independent")
        else:
            print("  overlap        : none — split IS subject-independent")
    return result


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

@dataclass
class ChannelStandardizer:
    """Per-channel zero-mean unit-variance scaling for (N, T, C) arrays.

    Statistics are pooled over both windows and timesteps, giving one mean and
    one std per channel. Fit on training subjects only.
    """

    mean_: np.ndarray | None = field(default=None)
    std_: np.ndarray | None = field(default=None)
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        self.mean_ = X.mean(axis=(0, 1)).astype(np.float32)
        self.std_ = X.std(axis=(0, 1)).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() on the training split first.")
        return ((X - self.mean_) / (self.std_ + self.eps)).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def subject_split(
    subjects: np.ndarray,
    val_subjects: tuple[int, ...] | None = DEFAULT_VAL_SUBJECTS,
    n_val: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Split window indices by volunteer, never by window.

    Returns (train_idx, val_idx, val_subjects_used).
    """
    uniq = np.unique(subjects)
    if val_subjects is None:
        rng = np.random.default_rng(seed)
        val_subjects = tuple(sorted(rng.choice(uniq, size=n_val, replace=False).tolist()))
    unknown = set(val_subjects) - set(uniq.tolist())
    if unknown:
        raise ValueError(f"Validation subjects {sorted(unknown)} are not in this split.")
    mask = np.isin(subjects, list(val_subjects))
    return np.where(~mask)[0], np.where(mask)[0], tuple(val_subjects)


def loso_folds(subjects: np.ndarray):
    """Yield (train_idx, test_idx, held_out_subject) for leave-one-subject-out.

    Combine train and test partitions first (see `load_all`) to get all 30
    volunteers. 30 folds x 3 models is an overnight job on a mid-range GPU and
    gives per-subject variance for the results table.
    """
    for s in np.unique(subjects):
        test_idx = np.where(subjects == s)[0]
        yield np.where(subjects != s)[0], test_idx, int(s)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def load_bundle(
    root: str | Path,
    val_subjects: tuple[int, ...] | None = DEFAULT_VAL_SUBJECTS,
    standardize: bool = True,
    with_features561: bool = False,
    verbose: bool = True,
) -> dict:
    """Load the full protocol: train / val / test, standardised, ready to use.

    Every model must call this with the same arguments.
    """
    root = _resolve_root(root)

    X_tr_all = load_signals(root, "train")
    y_tr_all = load_labels(root, "train")
    s_tr_all = load_subjects(root, "train")

    X_te = load_signals(root, "test")
    y_te = load_labels(root, "test")
    s_te = load_subjects(root, "test")

    tr_idx, val_idx, val_used = subject_split(s_tr_all, val_subjects=val_subjects)

    X_tr, y_tr, s_tr = X_tr_all[tr_idx], y_tr_all[tr_idx], s_tr_all[tr_idx]
    X_val, y_val, s_val = X_tr_all[val_idx], y_tr_all[val_idx], s_tr_all[val_idx]

    scaler = ChannelStandardizer()
    if standardize:
        # Fitted on training subjects only — not on train+val, not on test.
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_te = scaler.transform(X_te)

    bundle = {
        "X_train": X_tr, "y_train": y_tr, "subj_train": s_tr,
        "X_val": X_val, "y_val": y_val, "subj_val": s_val,
        "X_test": X_te, "y_test": y_te, "subj_test": s_te,
        "scaler": scaler,
        "val_subjects": val_used,
        "class_names": ACTIVITY_NAMES,
        "input_shape": (WINDOW_LEN, N_CHANNELS),
        "n_classes": N_CLASSES,
    }

    if with_features561:
        F_all = load_features561(root, "train")
        bundle["F_train"] = F_all[tr_idx]
        bundle["F_val"] = F_all[val_idx]
        bundle["F_test"] = load_features561(root, "test")

    if verbose:
        print(f"  train : {X_tr.shape}  subjects={len(np.unique(s_tr))}")
        print(f"  val   : {X_val.shape}  subjects={sorted(val_used)}")
        print(f"  test  : {X_te.shape}  subjects={len(np.unique(s_te))}")
        counts = np.bincount(y_tr, minlength=N_CLASSES)
        print("  train class balance:")
        for name, c in zip(ACTIVITY_NAMES, counts):
            print(f"    {name:<20} {c:5d}  ({100 * c / len(y_tr):.1f}%)")

    return bundle


def load_all(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate both partitions — all 30 subjects. For LOSO only.

    Returned data is UNSCALED: fit the standardiser inside each fold, or you
    leak the held-out subject's statistics into training.
    """
    X = np.concatenate([load_signals(root, "train"), load_signals(root, "test")])
    y = np.concatenate([load_labels(root, "train"), load_labels(root, "test")])
    s = np.concatenate([load_subjects(root, "train"), load_subjects(root, "test")])
    return X, y, s


# ---------------------------------------------------------------------------
# PyTorch glue (optional — import only if torch is installed)
# ---------------------------------------------------------------------------

def make_loaders(bundle: dict, batch_size: int = 128, channels_first: bool = False,
                 num_workers: int = 0):
    """Build train/val/test DataLoaders from a bundle.

    channels_first=True gives (N, 9, 128) for nn.Conv1d.
    Leave it False for nn.LSTM, which wants (N, 128, 9) with batch_first=True.
    The MLP flattens either way.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def _ds(X, y):
        t = torch.from_numpy(np.ascontiguousarray(X))
        if channels_first:
            t = t.permute(0, 2, 1).contiguous()
        return TensorDataset(t, torch.from_numpy(y))

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(_ds(bundle["X_train"], bundle["y_train"]), shuffle=True, **common),
        DataLoader(_ds(bundle["X_val"], bundle["y_val"]), shuffle=False, **common),
        DataLoader(_ds(bundle["X_test"], bundle["y_test"]), shuffle=False, **common),
    )


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity-check the UCI HAR dataset.")
    ap.add_argument("--root", default="data/UCI HAR Dataset",
                    help="Path to the extracted 'UCI HAR Dataset' folder.")
    args = ap.parse_args()

    print("\n[1] Subject independence check")
    res = verify_subject_independence(args.root)

    print("\n[2] Loading bundle under the shared protocol")
    b = load_bundle(args.root)

    print("\n[3] Leakage guards")
    ok_val = not (set(b["subj_train"].tolist()) & set(b["subj_val"].tolist()))
    ok_test = not (set(b["subj_train"].tolist()) & set(b["subj_test"].tolist()))
    print(f"  train/val  subject overlap : {'none' if ok_val else 'LEAK'}")
    print(f"  train/test subject overlap : {'none' if ok_test else 'LEAK'}")
    print(f"  scaler fitted on           : {len(np.unique(b['subj_train']))} subjects")

    all_ok = res["subject_independent"] and ok_val and ok_test
    print(f"\n  ==> protocol {'OK' if all_ok else 'FAILED'}\n")


if __name__ == "__main__":
    main()
