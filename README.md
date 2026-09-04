# HAR Model Comparison

A reproducible comparison of MLP, 1D-CNN, and BiLSTM models on the UCI Human Activity Recognition dataset.

## Project layout

- `src/har_data.py`: shared loading, subject splitting, standardisation, and reproducibility utilities.
- `src/models/`: model implementations owned by each member.
- `src/train.py`: shared training loop entry point.
- `src/evaluate.py`: shared metrics and confusion-matrix utilities.
- `notebooks/`: one experiment notebook per member.
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
