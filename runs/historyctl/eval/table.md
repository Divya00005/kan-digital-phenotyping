# History-only control: pooled cross-validation (170 train + validation students)

Out-of-fold predictions, 5 folds x 5 seeds; test set not used. 95% CI from a student-level bootstrap (1,000 resamples); differences are paired.

**Primary: screen-positive AUROC, averaged over depression and anxiety (higher is better).**

| Horizon | Mean score | Persistence | Linear/logistic on history | KAN version H (history only) | KAN version B (sensing + history) | B − H |
|---|---|---|---|---|---|---|
| 1w | 0.453 [0.378, 0.535] | 0.882 [0.860, 0.899] | 0.910 [0.890, 0.925] | 0.907 [0.887, 0.922] | 0.902 [0.880, 0.917] | -0.006 [-0.010, -0.002] |
| 2w | 0.458 [0.382, 0.539] | 0.852 [0.826, 0.873] | 0.886 [0.862, 0.904] | 0.885 [0.861, 0.903] | 0.881 [0.856, 0.900] | -0.004 [-0.008, -0.000] |
| 4w | 0.454 [0.382, 0.536] | 0.832 [0.803, 0.854] | 0.867 [0.840, 0.888] | 0.867 [0.837, 0.887] | 0.865 [0.836, 0.886] | -0.002 [-0.007, 0.004] |
| 8w | 0.458 [0.384, 0.541] | 0.801 [0.763, 0.831] | 0.840 [0.802, 0.867] | 0.842 [0.805, 0.869] | 0.840 [0.804, 0.867] | -0.002 [-0.007, 0.004] |

**Decision rule** (sensing adds ranking information if the B − H CI is entirely above 0 at ≥ 2 horizons): B better at weeks none, B worse at weeks [1, 2] → **sensing adds ranking information: NO**

**Onset AUC-PR, averaged over tasks (higher is better)**

| Horizon | Mean score | Persistence | Linear/logistic on history | KAN version H (history only) | KAN version B (sensing + history) | B − H |
|---|---|---|---|---|---|---|
| 1w | 0.051 [0.039, 0.068] | 0.142 [0.117, 0.171] | 0.270 [0.225, 0.314] | 0.266 [0.221, 0.312] | 0.255 [0.211, 0.298] | -0.011 [-0.024, 0.003] |
| 2w | 0.059 [0.045, 0.078] | 0.150 [0.123, 0.179] | 0.283 [0.232, 0.331] | 0.279 [0.232, 0.325] | 0.267 [0.223, 0.311] | -0.012 [-0.024, 0.002] |
| 4w | 0.061 [0.048, 0.082] | 0.146 [0.117, 0.178] | 0.276 [0.225, 0.327] | 0.272 [0.221, 0.322] | 0.267 [0.213, 0.318] | -0.005 [-0.018, 0.008] |
| 8w | 0.071 [0.055, 0.096] | 0.146 [0.114, 0.182] | 0.257 [0.204, 0.313] | 0.260 [0.201, 0.317] | 0.255 [0.198, 0.311] | -0.005 [-0.016, 0.007] |

**AUC-PR, averaged over tasks (higher is better)**

| Horizon | Mean score | Persistence | Linear/logistic on history | KAN version H (history only) | KAN version B (sensing + history) | B − H |
|---|---|---|---|---|---|---|
| 1w | 0.105 [0.077, 0.145] | 0.519 [0.430, 0.594] | 0.639 [0.546, 0.709] | 0.618 [0.525, 0.689] | 0.592 [0.507, 0.660] | -0.025 [-0.036, -0.010] |
| 2w | 0.107 [0.078, 0.150] | 0.452 [0.356, 0.534] | 0.580 [0.471, 0.662] | 0.562 [0.459, 0.643] | 0.545 [0.449, 0.623] | -0.017 [-0.029, 0.001] |
| 4w | 0.104 [0.076, 0.144] | 0.403 [0.308, 0.488] | 0.521 [0.406, 0.610] | 0.507 [0.400, 0.590] | 0.500 [0.395, 0.578] | -0.007 [-0.022, 0.007] |
| 8w | 0.109 [0.079, 0.151] | 0.372 [0.269, 0.466] | 0.487 [0.360, 0.584] | 0.474 [0.354, 0.570] | 0.463 [0.350, 0.554] | -0.011 [-0.024, 0.003] |

**MAE, averaged over tasks (lower is better)**

| Horizon | Mean score | Persistence | Linear/logistic on history | KAN version H (history only) | KAN version B (sensing + history) | B − H |
|---|---|---|---|---|---|---|
| 1w | 1.095 [1.031, 1.168] | 0.559 [0.506, 0.617] | 0.600 [0.556, 0.648] | 0.628 [0.578, 0.682] | 0.671 [0.621, 0.725] | 0.043 [0.033, 0.053] |
| 2w | 1.100 [1.035, 1.175] | 0.644 [0.583, 0.710] | 0.678 [0.630, 0.730] | 0.685 [0.631, 0.742] | 0.718 [0.664, 0.776] | 0.033 [0.022, 0.042] |
| 4w | 1.097 [1.030, 1.171] | 0.711 [0.645, 0.784] | 0.740 [0.692, 0.793] | 0.736 [0.682, 0.794] | 0.754 [0.699, 0.812] | 0.018 [0.008, 0.026] |
| 8w | 1.108 [1.037, 1.186] | 0.781 [0.708, 0.862] | 0.796 [0.744, 0.852] | 0.793 [0.734, 0.854] | 0.807 [0.749, 0.867] | 0.014 [0.005, 0.022] |

**Paired differences in AUROC (avg over tasks)**

| Horizon | KAN version B (sensing + history) − KAN version H (history only) | KAN version B (sensing + history) − Persistence | KAN version H (history only) − Persistence | Linear/logistic on history − Persistence | KAN version B (sensing + history) − Linear/logistic on history | KAN version H (history only) − Linear/logistic on history |
|---|---|---|---|---|---|---|
| 1w | -0.006 [-0.010, -0.002] | 0.020 [0.013, 0.026] | 0.025 [0.020, 0.031] | 0.028 [0.023, 0.033] | -0.008 [-0.013, -0.003] | -0.003 [-0.006, 0.000] |
| 2w | -0.004 [-0.008, -0.000] | 0.029 [0.023, 0.036] | 0.033 [0.027, 0.040] | 0.034 [0.028, 0.040] | -0.005 [-0.010, -0.000] | -0.001 [-0.004, 0.002] |
| 4w | -0.002 [-0.007, 0.004] | 0.033 [0.026, 0.040] | 0.034 [0.028, 0.042] | 0.035 [0.026, 0.044] | -0.002 [-0.010, 0.006] | -0.001 [-0.006, 0.005] |
| 8w | -0.002 [-0.007, 0.004] | 0.039 [0.031, 0.049] | 0.041 [0.033, 0.050] | 0.039 [0.028, 0.049] | 0.001 [-0.009, 0.012] | 0.002 [-0.005, 0.012] |

**Spread of AUROC (avg over tasks): mean ± SD over seeds (pooled folds) and over folds (seed-averaged)**

| Horizon | H over seeds | B over seeds | H over folds | B over folds |
|---|---|---|---|---|
| 1w | 0.905 ± 0.002 | 0.896 ± 0.004 | 0.906 ± 0.019 | 0.901 ± 0.015 |
| 2w | 0.883 ± 0.003 | 0.876 ± 0.004 | 0.883 ± 0.022 | 0.880 ± 0.019 |
| 4w | 0.864 ± 0.002 | 0.860 ± 0.004 | 0.864 ± 0.020 | 0.864 ± 0.016 |
| 8w | 0.840 ± 0.003 | 0.835 ± 0.004 | 0.841 ± 0.027 | 0.840 ± 0.026 |

Caveat (pre-registered): inputs were normalised with the original 149 training students, so some held-out students' inputs contributed to those means / SDs (inputs only, no labels; same for B and H).
