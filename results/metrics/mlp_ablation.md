# MLP ablation (3 seeds per variant, mean ± std)

One factor varied at a time around the base configuration, across both input representations. Ordered as run; 'Val loss rank' ranks by mean validation loss (the selection criterion). Test columns are reported, not used for selection.

| Variant | Representation | Params | Val loss | Val loss rank | Test accuracy | Test macro-F1 | Best epoch | s/run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw: base (h=256/128, p=0.3) | raw 128 x 9 window | 329,606 | 0.4481 ± 0.0029 | 6 | 0.8971 ± 0.0052 | 0.8974 ± 0.0049 | 6.7 | 6 |
| raw: hidden 64/32 (quarter) | raw 128 x 9 window | 76,262 | 0.4495 ± 0.0152 | 8 | 0.8918 ± 0.0009 | 0.8922 ± 0.0015 | 19.7 | 7 |
| raw: hidden 128/64 (half) | raw 128 x 9 window | 156,614 | 0.4441 ± 0.0167 | 5 | 0.8980 ± 0.0029 | 0.8983 ± 0.0029 | 10.7 | 6 |
| raw: hidden 512/256 (double) | raw 128 x 9 window | 724,742 | 0.4747 ± 0.0046 | 9 | 0.8903 ± 0.0092 | 0.8904 ± 0.0092 | 3.0 | 7 |
| raw: dropout 0.0 | raw 128 x 9 window | 329,606 | 0.4490 ± 0.0294 | 7 | 0.8944 ± 0.0073 | 0.8952 ± 0.0071 | 3.0 | 4 |
| raw: dropout 0.5 | raw 128 x 9 window | 329,606 | 0.4282 ± 0.0063 | 4 | 0.8961 ± 0.0021 | 0.8960 ± 0.0020 | 8.7 | 7 |
| 561-feat: base (h=256/128, p=0.3) | 561 hand-crafted features | 178,310 | 0.2414 ± 0.0157 | 1 | 0.9204 ± 0.0067 | 0.9217 ± 0.0068 | 9.3 | 6 |
| 561-feat: hidden 128/64 (half) | 561 hand-crafted features | 80,966 | 0.2565 ± 0.0320 | 3 | 0.9139 ± 0.0187 | 0.9142 ± 0.0184 | 15.3 | 6 |
| 561-feat: dropout 0.5 | 561 hand-crafted features | 178,310 | 0.2491 ± 0.0120 | 2 | 0.9152 ± 0.0082 | 0.9163 ± 0.0085 | 14.7 | 8 |

## Read-out

- Capacity sweep (raw input, 4 widths from 64/32 to 512/256): test accuracy spans 0.8903 to 0.8980, a range of 0.0077. A span far smaller than the gap to the 1D-CNN means the baseline is not capacity-limited, so that gap is attributable to inductive bias rather than to model size.
- Best raw-input variant 0.8980 vs best 561-feature variant 0.9204: a difference of +0.0224 from changing only the input representation, with the architecture and optimiser settings held identical.
- Selected by validation loss: 561-feat: base (h=256/128, p=0.3).
