# Multi-seed results (raw 128 x 9 input, test set, 9 unseen subjects)

Validation subjects fixed at [8, 16, 23, 29]; only the seed varies (weight initialisation and batch order). Values are mean ± sample std.

| Model | Seeds | Params | Accuracy | Macro-F1 | Best epoch | Sitting→standing | Standing→sitting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1D-CNN | 5 | 128,646 | 0.9211 ± 0.0065 | 0.9212 ± 0.0065 | 4.8 | 0.170 ± 0.022 | 0.114 ± 0.014 |
| MLP (raw signals) | 5 | 329,606 | 0.8925 ± 0.0087 | 0.8930 ± 0.0084 | 5.2 | 0.177 ± 0.023 | 0.155 ± 0.028 |
| MLP (561 features) | 5 | 178,310 | 0.9170 ± 0.0067 | 0.9178 ± 0.0072 | 12.8 | 0.221 ± 0.041 | 0.032 ± 0.014 |

## Per-class F1 (mean ± std)

| Model | WALKING | WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | SITTING | STANDING | LAYING |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1D-CNN | 0.978 ± 0.010 | 0.919 ± 0.012 | 0.939 ± 0.016 | 0.841 ± 0.009 | 0.868 ± 0.005 | 0.981 ± 0.011 |
| MLP (raw signals) | 0.932 ± 0.008 | 0.894 ± 0.007 | 0.913 ± 0.008 | 0.804 ± 0.022 | 0.840 ± 0.019 | 0.974 ± 0.000 |
| MLP (561 features) | 0.934 ± 0.026 | 0.905 ± 0.026 | 0.969 ± 0.009 | 0.852 ± 0.021 | 0.871 ± 0.003 | 0.975 ± 0.013 |
