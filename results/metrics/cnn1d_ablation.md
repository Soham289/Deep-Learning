# 1D-CNN ablation (3 seeds per variant, mean ± std)

One factor varied at a time around the base configuration. Ordered as run; 'Val loss rank' ranks by mean validation loss (the selection criterion). Test columns are reported, not used for selection.

| Variant | Params | Val loss | Val loss rank | Test accuracy | Test macro-F1 | Best epoch | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base (k=7/5/5, w=64/128/128, p=0.3) | 128,646 | 0.4130 ± 0.0136 | 2 | 0.9234 ± 0.0057 | 0.9235 ± 0.0056 | 5.3 | 90 |
| kernels 3/3/3 | 77,190 | 0.4233 ± 0.0147 | 6 | 0.9191 ± 0.0038 | 0.9189 ± 0.0036 | 8.7 | 84 |
| kernels 5/5/5 | 127,494 | 0.4065 ± 0.0106 | 1 | 0.9187 ± 0.0064 | 0.9193 ± 0.0061 | 3.3 | 79 |
| kernels 11/7/7 | 180,102 | 0.4227 ± 0.0170 | 5 | 0.9258 ± 0.0112 | 0.9258 ± 0.0113 | 6.7 | 116 |
| width 32/64/64 (half) | 33,606 | 0.4259 ± 0.0307 | 8 | 0.9230 ± 0.0006 | 0.9231 ± 0.0009 | 19.3 | 66 |
| width 128/256/256 (double) | 503,046 | 0.4149 ± 0.0120 | 3 | 0.9164 ± 0.0068 | 0.9171 ± 0.0064 | 2.0 | 229 |
| dropout 0.0 | 128,646 | 0.4242 ± 0.0183 | 7 | 0.9266 ± 0.0048 | 0.9267 ± 0.0049 | 5.7 | 93 |
| dropout 0.5 | 128,646 | 0.4156 ± 0.0104 | 4 | 0.9196 ± 0.0028 | 0.9198 ± 0.0030 | 5.3 | 90 |
