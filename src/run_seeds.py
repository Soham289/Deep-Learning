"""Multi-seed runner: trains one model at several seeds, reports mean +/- std.

Every figure in the first results table came from a single run at seed 42, so
differences of a point or two between models cannot be told apart from seed
noise. This repeats each model with the validation subjects held fixed (so the
data is identical) while the seed changes weight initialisation and batch order.

    python src/run_seeds.py --model cnn1d            # seeds 42..46
    python src/run_seeds.py --model bilstm --seeds 1 2 3
    python src/run_seeds.py --summarize              # cross-model table + plot

Writes results/metrics/<model>_seeds.json; --summarize writes
results/metrics/seeds_summary.md and results/figures/seeds_accuracy.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore", message=".*pin_memory.*")

from experiments import SPECS, aggregate, run_once  # noqa: E402
from har_data import load_bundle  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def run_model(name: str, seeds: list[int], max_epochs: int | None = None) -> dict:
    bundle = load_bundle(DATA_ROOT, verbose=False)
    train_kwargs = {"max_epochs": max_epochs} if max_epochs else None
    records = []
    for seed in seeds:
        r = run_once(name, bundle, seed, train_kwargs=train_kwargs)
        records.append(r)
        print(f"  seed {seed}: acc {r['test_accuracy']:.4f}  macroF1 {r['test_macro_f1']:.4f}  "
              f"best epoch {r['best_epoch']}  {r['training_time_sec']:.0f}s")

    payload = {
        "model": SPECS[name]["label"],
        "key": name,
        "val_subjects": list(bundle["val_subjects"]),
        "seeds": seeds,
        "train_config": {**SPECS[name]["train"], **(train_kwargs or {})},
        "aggregate": aggregate(records),
        "runs": records,
    }
    agg = payload["aggregate"]
    print(f"  => acc {agg['test_accuracy']['mean']:.4f} +/- {agg['test_accuracy']['std']:.4f} | "
          f"macroF1 {agg['test_macro_f1']['mean']:.4f} +/- {agg['test_macro_f1']['std']:.4f}")
    return payload


def summarize() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payloads = []
    for key in SPECS:
        p = METRICS_DIR / f"{key}_seeds.json"
        if p.exists():
            payloads.append(json.loads(p.read_text()))
    if not payloads:
        raise SystemExit("No *_seeds.json files found; run --model first.")

    def pm(s, digits=4):
        return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"

    lines = [
        "# Multi-seed results (raw 128 x 9 input, test set, 9 unseen subjects)",
        "",
        f"Validation subjects fixed at {payloads[0]['val_subjects']}; only the seed varies "
        "(weight initialisation and batch order). Values are mean ± sample std.",
        "",
        "| Model | Seeds | Params | Accuracy | Macro-F1 | Best epoch | Sitting→standing | Standing→sitting |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in payloads:
        a = p["aggregate"]
        lines.append(
            f"| {p['model']} | {a['n_runs']} | {int(a['n_params']['mean']):,} | "
            f"{pm(a['test_accuracy'])} | {pm(a['test_macro_f1'])} | "
            f"{a['best_epoch']['mean']:.1f} | {pm(a['sitting_as_standing'], 3)} | "
            f"{pm(a['standing_as_sitting'], 3)} |")

    classes = list(payloads[0]["aggregate"]["per_class_f1"])
    lines += ["", "## Per-class F1 (mean ± std)", "",
              "| Model | " + " | ".join(classes) + " |",
              "| --- | " + " | ".join(["---:"] * len(classes)) + " |"]
    for p in payloads:
        pc = p["aggregate"]["per_class_f1"]
        lines.append(f"| {p['model']} | " + " | ".join(pm(pc[c], 3) for c in classes) + " |")

    out_md = METRICS_DIR / "seeds_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved -> {out_md.relative_to(PROJECT_ROOT)}")

    if len(payloads) >= 2:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for i, p in enumerate(payloads):
            accs = [r["test_accuracy"] for r in p["runs"]]
            ax.scatter([i] * len(accs), accs, alpha=0.7, s=36)
            m = sum(accs) / len(accs)
            ax.hlines(m, i - 0.25, i + 0.25, color="black", linewidth=2)
        ax.set_xticks(range(len(payloads)))
        ax.set_xticklabels([p["model"] for p in payloads], rotation=10)
        ax.set_ylabel("test accuracy")
        ax.set_title("Test accuracy across seeds (bar = mean)")
        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "seeds_accuracy.png", dpi=150)
        plt.close(fig)
        print("saved -> results/figures/seeds_accuracy.png")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=list(SPECS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--max-epochs", type=int, default=None, help="cap epochs (smoke tests)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return
    if not args.model:
        ap.error("--model is required unless --summarize is given")

    print(f"{SPECS[args.model]['label']} x seeds {args.seeds}")
    payload = run_model(args.model, args.seeds, args.max_epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.model}_seeds.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
