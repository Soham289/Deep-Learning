# HAR Model Comparison

A reproducible comparison of MLP, 1D-CNN, and BiLSTM models on the UCI Human Activity Recognition dataset.

## Project layout

- `src/har_data.py`: shared loading, subject splitting, standardisation, and reproducibility utilities.
- `src/models/`: model implementations owned by each member.
- `src/train.py`: shared training loop entry point.
- `src/evaluate.py`: shared metrics and confusion-matrix utilities.
- `src/experiments.py`: each raw-signal model's build and training config, plus single-run and aggregation helpers.
- `src/run_seeds.py`: multi-seed runner and cross-model summary table.
- `src/run_loso.py`: leave-one-subject-out cross-validation.
- `notebooks/`: one experiment notebook per member, plus standalone experiment scripts.
- `results/`: committed metrics and figures.
- `data/`: dataset download instructions only; the dataset itself is ignored.
- `docs/`: synopsis, interim report, and final report.

## Setup

Install PyTorch for the target hardware first, then install the remaining dependencies:

```text
pip install torch
pip install -r requirements.txt
```

Download and extract the UCI HAR Dataset as described in `data/README.md`, then run the shared sanity check from the project root:

```text
python src/har_data.py --root "data/UCI HAR Dataset"
```

## Reproducing the multi-run experiments

Run from the project root. Each model keeps the hyperparameters its owner settled on (see `src/experiments.py`); the validation subjects stay fixed at 8, 16, 23 and 29.

```text
python src/run_seeds.py --model cnn1d          # 5 seeds -> results/metrics/cnn1d_seeds.json
python src/run_seeds.py --model mlp
python src/run_seeds.py --model bilstm         # slow on CPU (~40 s/epoch); fast on a GPU
python src/run_seeds.py --summarize            # mean +/- std table and plot across models
python src/run_loso.py --model cnn1d           # 30-fold leave-one-subject-out, resumable
python notebooks/cnn1d_validation_noise_study.py
python notebooks/cnn1d_ablation.py
```

Training times in the metric files depend on the machine, so only compare them between runs made on the same hardware.
