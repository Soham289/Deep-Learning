"""1D-CNN ablation: which structural choices matter? (Member B)

Varies one thing at a time around the base configuration (kernels 7/5/5,
widths 64/128/128, dropout 0.3) to support the discussion of *why* the
convolutional model does well: kernel size (how much local context each filter
sees), channel width (capacity), and dropout (regularisation).

Each variant is trained at three seeds with everything else identical. Variants
are ranked by mean VALIDATION loss -- test accuracy is reported alongside but is
never used to choose, so the ablation does not tune on the test set.

    python notebooks/cnn1d_ablation.py

Writes results/metrics/cnn1d_ablation.json and cnn1d_ablation.md.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from experiments import aggregate, run_once  # noqa: E402
from har_data import load_bundle  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
SEEDS = [42, 43, 44]

BASE = dict(channel_sizes=(64, 128, 128), kernel_sizes=(7, 5, 5), dropout=0.3)
VARIANTS = [
    ("base (k=7/5/5, w=64/128/128, p=0.3)", {}),
    ("kernels 3/3/3", dict(kernel_sizes=(3, 3, 3))),
    ("kernels 5/5/5", dict(kernel_sizes=(5, 5, 5))),
    ("kernels 11/7/7", dict(kernel_sizes=(11, 7, 7))),
    ("width 32/64/64 (half)", dict(channel_sizes=(32, 64, 64))),
    ("width 128/256/256 (double)", dict(channel_sizes=(128, 256, 256))),
    ("dropout 0.0", dict(dropout=0.0)),
    ("dropout 0.5", dict(dropout=0.5)),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    bundle = load_bundle(DATA_ROOT, verbose=False)
    results = []
    for label, override in VARIANTS:
        kwargs = {**BASE, **override}
        runs = []
        for seed in SEEDS:
            r = run_once("cnn1d", bundle, seed, model_kwargs=kwargs)
            runs.append(r)
        agg = aggregate(runs)
        results.append({"variant": label, "model_kwargs": {k: list(v) if isinstance(v, tuple) else v
                                                           for k, v in kwargs.items()},
                        "aggregate": agg, "runs": runs})
        print(f"{label:40s} params {int(agg['n_params']['mean']):>8,}  "
              f"val loss {agg['best_val_loss']['mean']:.4f}  "
              f"test acc {agg['test_accuracy']['mean']:.4f} +/- {agg['test_accuracy']['std']:.4f}  "
              f"{agg['training_time_sec']['mean']:.0f}s/run", flush=True)

    ranked = sorted(results, key=lambda r: r["aggregate"]["best_val_loss"]["mean"])
    (METRICS_DIR / "cnn1d_ablation.json").write_text(
        json.dumps({"seeds": SEEDS, "base": {k: list(v) if isinstance(v, tuple) else v
                                             for k, v in BASE.items()},
                    "selected_by_validation_loss": ranked[0]["variant"],
                    "variants": results}, indent=2))

    lines = [
        "# 1D-CNN ablation (3 seeds per variant, mean ± std)",
        "",
        "One factor varied at a time around the base configuration. Ordered as run; "
        "'Val loss rank' ranks by mean validation loss (the selection criterion). "
        "Test columns are reported, not used for selection.",
        "",
        "| Variant | Params | Val loss | Val loss rank | Test accuracy | Test macro-F1 | Best epoch | s/run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rank = {r["variant"]: i + 1 for i, r in enumerate(ranked)}
    for r in results:
        a = r["aggregate"]
        lines.append(
            f"| {r['variant']} | {int(a['n_params']['mean']):,} | "
            f"{a['best_val_loss']['mean']:.4f} ± {a['best_val_loss']['std']:.4f} | {rank[r['variant']]} | "
            f"{a['test_accuracy']['mean']:.4f} ± {a['test_accuracy']['std']:.4f} | "
            f"{a['test_macro_f1']['mean']:.4f} ± {a['test_macro_f1']['std']:.4f} | "
            f"{a['best_epoch']['mean']:.1f} | {a['training_time_sec']['mean']:.0f} |")
    (METRICS_DIR / "cnn1d_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nselected by validation loss: {ranked[0]['variant']}")


if __name__ == "__main__":
    main()
