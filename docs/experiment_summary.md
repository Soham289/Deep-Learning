# Experiment Results Summary

## Protocol

The runs use seed 42 and validation subjects 8, 16, 23, and 29. The official UCI test partition contains 2,947 windows from nine subjects. The raw-signal MLP, CNN, and BiLSTM consume the same 128-by-9 inertial windows; validation loss selects the checkpoint, and the reported metrics are measured on the test partition.

## Results

| Model | Input | Parameters | Best epoch | Test accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| MLP | Raw 128 x 9 signals | 329,606 | 6 | 0.8996 | 0.9001 |
| 1D-CNN | Raw 128 x 9 signals | 128,646 | 6 | 0.9298 | 0.9296 |
| BiLSTM (hidden size 96) | Raw 128 x 9 signals | 306,054 | 5 | 0.8890 | 0.8883 |

The MLP's additional 561-feature experiment achieved test accuracy 0.9253 and macro-F1 0.9266 with 178,310 parameters. This uses the dataset's precomputed features, so it is a separate representation comparison and should not be treated as a raw-input architecture result.

Among the recorded raw-input runs, the CNN has the highest test accuracy and macro-F1. The 561-feature MLP approaches the CNN result, while outperforming the raw-input MLP in this run. Sitting versus standing remains a notable confusion for all three raw-input models.

These are single seeded runs, not estimates of variance or statistical significance. Training times are recorded in each metric JSON but are omitted from this cross-model table because hardware and runtime conditions should be verified before comparing them.

## Result Artifacts

- Raw MLP: `results/metrics/mlp_metrics.json`
- 561-feature MLP: `results/metrics/mlp561_metrics.json`
- CNN: `results/metrics/cnn1d_metrics.json`
- BiLSTM: recorded in `results/metrics/bilstm_96_metrics.json` on the `avni-bilstm` branch
