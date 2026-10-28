"""MLP ablation: what does the baseline's capacity actually buy? (Member A)

The MLP is the control in this comparison: it flattens the window and so
encodes no temporal or spatial prior. That makes two questions worth asking,
and neither can be answered by the architecture comparison alone.

  1. Is the baseline's result a function of its capacity? If widening or
     narrowing the hidden layers barely moves it, then the gap to the 1D-CNN is
     about inductive bias rather than about the MLP being too small -- which is
     the claim the whole comparison rests on.
  2. How much of that gap does the hand-crafted 561-feature representation
     close, under identical architecture and identical optimiser settings?

Every variant is trained at three seeds with everything else held fixed, and
variants are ranked by mean VALIDATION loss. Test accuracy is reported beside
it but never used to choose, so nothing here tunes on the test set.

    python notebooks/mlp_ablation.py

Writes results/metrics/mlp_ablation.json and mlp_ablation.md.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from experiments import aggregate, bundle_kwargs, run_once  # noqa: E402
from har_data import load_bundle  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
SEEDS = [42, 43, 44]

BASE = dict(hidden_sizes=(256, 128), dropout=0.3)

#: (label, model key, model_kwargs override). The model key selects the input
#: representation: "mlp" is the raw 1,152-dim flattened window, "mlp561" is the
#: 561 hand-crafted features. Both use the identical HARMlp class and the
#: identical training configuration.
VARIANTS = [
    ("raw: base (h=256/128, p=0.3)",      "mlp",    {}),
    ("raw: hidden 64/32 (quarter)",       "mlp",    dict(hidden_sizes=(64, 32))),
    ("raw: hidden 128/64 (half)",         "mlp",    dict(hidden_sizes=(128, 64))),
    ("raw: hidden 512/256 (double)",      "mlp",    dict(hidden_sizes=(512, 256))),
    ("raw: dropout 0.0",                  "mlp",    dict(dropout=0.0)),
    ("raw: dropout 0.5",                  "mlp",    dict(dropout=0.5)),
    ("561-feat: base (h=256/128, p=0.3)", "mlp561", {}),
    ("561-feat: hidden 128/64 (half)",    "mlp561", dict(hidden_sizes=(128, 64))),
    ("561-feat: dropout 0.5",             "mlp561", dict(dropout=0.5)),
]


def _jsonable(kwargs: dict) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    # One bundle with the feature arrays loaded, reused by both representations.
    bundle = load_bundle(DATA_ROOT, verbose=False, **bundle_kwargs("mlp561"))

    results = []
    for label, key, override in VARIANTS:
        kwargs = {**BASE, **override}
        runs = [run_once(key, bundle, seed, model_kwargs=kwargs) for seed in SEEDS]
        agg = aggregate(runs)
        results.append({
            "variant": label,
            "model_key": key,
            "representation": "561 hand-crafted features" if key == "mlp561"
                              else "raw 128 x 9 window",
            "model_kwargs": _jsonable(kwargs),
            "aggregate": agg,
            "runs": runs,
        })
        print(f"{label:38s} params {int(agg['n_params']['mean']):>8,}  "
              f"val loss {agg['best_val_loss']['mean']:.4f}  "
              f"test acc {agg['test_accuracy']['mean']:.4f} +/- {agg['test_accuracy']['std']:.4f}  "
              f"{agg['training_time_sec']['mean']:.0f}s/run", flush=True)

    ranked = sorted(results, key=lambda r: r["aggregate"]["best_val_loss"]["mean"])
    rank = {r["variant"]: i + 1 for i, r in enumerate(ranked)}

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "mlp_ablation.json").write_text(json.dumps({
        "seeds": SEEDS,
        "base": _jsonable(BASE),
        "selected_by_validation_loss": ranked[0]["variant"],
        "variants": results,
    }, indent=2))

    lines = [
        "# MLP ablation (3 seeds per variant, mean ± std)",
        "",
        "One factor varied at a time around the base configuration, across both input "
        "representations. Ordered as run; 'Val loss rank' ranks by mean validation loss "
        "(the selection criterion). Test columns are reported, not used for selection.",
        "",
        "| Variant | Representation | Params | Val loss | Val loss rank | Test accuracy | "
        "Test macro-F1 | Best epoch | s/run |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        a = r["aggregate"]
        lines.append(
            f"| {r['variant']} | {r['representation']} | {int(a['n_params']['mean']):,} | "
            f"{a['best_val_loss']['mean']:.4f} ± {a['best_val_loss']['std']:.4f} | "
            f"{rank[r['variant']]} | "
            f"{a['test_accuracy']['mean']:.4f} ± {a['test_accuracy']['std']:.4f} | "
            f"{a['test_macro_f1']['mean']:.4f} ± {a['test_macro_f1']['std']:.4f} | "
            f"{a['best_epoch']['mean']:.1f} | {a['training_time_sec']['mean']:.0f} |")

    # Capacity sensitivity: base plus the three width variants, dropout excluded,
    # so the span below reflects model size alone. This is the number the
    # discussion needs to rule out "the baseline was simply too small".
    capacity = [r for r in results
                if r["model_key"] == "mlp" and "dropout" not in r["variant"]]
    raw_acc = [r["aggregate"]["test_accuracy"]["mean"] for r in capacity]

    raw_all = [r["aggregate"]["test_accuracy"]["mean"] for r in results
               if r["model_key"] == "mlp"]
    feat_all = [r["aggregate"]["test_accuracy"]["mean"] for r in results
                if r["model_key"] == "mlp561"]
    best_raw, best_feat = max(raw_all), max(feat_all)

    lines += [
        "",
        "## Read-out",
        "",
        f"- Capacity sweep (raw input, {len(capacity)} widths from 64/32 to 512/256): test accuracy "
        f"spans {min(raw_acc):.4f} to {max(raw_acc):.4f}, a range of {max(raw_acc) - min(raw_acc):.4f}. "
        "A span far smaller than the gap to the 1D-CNN means the baseline is not capacity-limited, "
        "so that gap is attributable to inductive bias rather than to model size.",
        f"- Best raw-input variant {best_raw:.4f} vs best 561-feature variant {best_feat:.4f}: "
        f"a difference of {best_feat - best_raw:+.4f} from changing only the input representation, "
        "with the architecture and optimiser settings held identical.",
        f"- Selected by validation loss: {ranked[0]['variant']}.",
    ]

    (METRICS_DIR / "mlp_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nselected by validation loss: {ranked[0]['variant']}")


if __name__ == "__main__":
    main()
