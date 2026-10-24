# 02 · Neural networks, KANs, and the KAN-MTLNet reproduction

> Code: `kan.py` (`KANLinear`, `KANConv1d`), `model.py` (`KANMTLNet`, `focal_loss`,
> `multitask_loss`), `train.py`, tests `test_kan.py`.

## 1. From a neuron to an MLP

**Why we need it:** we want a function that turns inputs (e.g. 60 days of phone features) into an
output (e.g. "probability of screen-positive"), and we want to *learn* that function from data.

**A neuron** takes inputs x₁…xₙ, multiplies each by a weight, adds them up with a bias, and passes
the result through a fixed **activation function** σ:

```
y = σ( w1*x1 + w2*x2 + ... + wn*xn + b )
```

- **Everyday example:** deciding whether to take an umbrella. Inputs: clouds (x₁), forecast (x₂).
  Weights say how much each matters; the activation turns "evidence" into a yes/no-ish answer.
- **Activation functions** add bends. Without them, stacking layers is still just one straight-line
  (linear) function. Common ones: ReLU = max(0, x); **SiLU** = x · sigmoid(x) (smooth, used in KAN's
  "base" path); sigmoid = 1 / (1 + e^(−x)) squeezes to 0..1 (used to turn logits into probabilities).
- **MLP (multi-layer perceptron):** layers of neurons. Learnable parts = **weights on the
  connections**; the bend (activation) is **fixed** and sits on the **nodes**.
- **Training:** compute a loss (how wrong), compute gradients by backpropagation, and move weights a
  little downhill (here with the **AdamW** optimiser). One pass over the training data = one **epoch**.

## 2. The Kolmogorov–Arnold idea and KANs

**Why:** in an MLP every connection is just a number (a weight); all non-linearity comes from one
fixed function. A **KAN** (Kolmogorov–Arnold Network) puts a **learnable 1-D curve on every
connection** instead, and the nodes just add.

```
MLP edge:  x  --(w)-->  w*x            node:  sigma(sum)
KAN edge:  x  --phi()-->  phi(x)       node:  sum           (phi is learned, any smooth shape)
```

**Maths intuition (Kolmogorov–Arnold theorem):** any continuous function of several variables can
be written using only sums and functions of **one** variable. KANs use this as inspiration: layers
of "learn a 1-D curve per input, then add".

**Everyday example:** predicting sleep quality from caffeine and exercise. An MLP mixes them with
weights then bends the total. A KAN can learn "caffeine hurts more and more after 2 cups" (a curve
for caffeine) and "exercise helps up to an hour, then flattens" (a curve for exercise), then adds.
Each curve can be **plotted**, which is why KANs are called interpretable.

**Common mistakes:** (1) believing KANs are always more accurate — they are not, and in this project
a linear model matched them; (2) forgetting that each edge has many parameters, so KANs are bigger
than MLPs of the same width.

## 3. B-splines: how a learnable curve is stored

**Why:** we need a curve shape that a computer can learn by gradient descent. Store it as a
**weighted sum of fixed bumps** (basis functions); learning = adjusting the weights.

```
phi(x) = c1*B1(x) + c2*B2(x) + ... + c8*B8(x)        (B_k fixed bumps, c_k learned)
```

- **B-spline basis functions** are smooth bumps placed along a grid. With **cubic** splines
  (order k = 3) each bump covers 4 grid intervals; at any x only a few bumps are non-zero.
- **Key property (tested):** inside the grid, the bumps **add up to exactly 1** ("partition of
  unity"). `test_kan.py → test_partition_of_unity` checks this on 500 points in [-2.99, 2.99].

```
grid (G = 5 intervals on [-3, 3], plus 3 extra knots each side for cubic splines):
  -6.6 -5.4 -4.2 |-3  -1.8  -0.6  0.6  1.8  3| 4.2  5.4  6.6
  bumps:  /\  /\  /\  /\  /\  /\  /\  /\       -> 8 basis functions (G + k = 5 + 3)
```

**How the bumps are computed (Cox–de Boor recursion, `KANLinear.b_splines`):** order-0 bumps are
1 inside one interval and 0 elsewhere; each higher order blends two neighbouring lower-order bumps
with straight-line weights. After 3 rounds you get smooth cubic bumps.

## 4. `KANLinear`, line by line (`kan.py`)

```python
def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0), ...):
    h = (grid_range[1] - grid_range[0]) / grid_size                 # interval width 1.2
    knots = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]   # 12 knots
    self.register_buffer("grid", knots.expand(in_features, -1))     # same grid for every input
    n_basis = grid_size + spline_order                               # 8 bumps per edge
    self.base_weight   = Parameter(out, in)          # w_{j,i}
    self.spline_weight = Parameter(out, in, n_basis) # c_{j,i,k}
    self.bias          = Parameter(out)              # b_j
```

```python
def forward(self, x):
    base   = F.linear(F.silu(x), self.base_weight)                   # sum_i w_ji * SiLU(x_i)
    spline = F.linear(self.b_splines(x).reshape(N, -1),
                      self.spline_weight.reshape(out, -1))           # sum_i sum_k c_jik * B_k(x_i)
    return base + spline + self.bias
```

So every edge is `phi_{j,i}(x) = sum_k c_{j,i,k} B_k(x) + w_{j,i} * SiLU(x)` (paper Eq. 1–2), and
output j adds the curves of all inputs plus a bias.

- **Parameters per edge:** 8 spline coefficients + 1 base weight = 9. A `KANLinear(72, 16)` has
  72 × 16 × 9 + 16 = 10,384 parameters (an MLP layer of the same size has 1,168).
- **Why the SiLU "base" path:** it gives a sensible shape outside the grid, where all spline bumps
  are zero.
- **Why LayerNorm before KAN layers (in `model.py` / `forecast_model.py`):** the spline grid covers
  [-3, 3]; LayerNorm keeps hidden values roughly in that range so the curves are actually used.
- `edge_curve(j, i, xs)` returns one learned curve for plotting (`test_edge_decomposition` checks that
  the layer output equals the sum of its edge curves plus bias).
- `test_fits_1d_curve`: one KAN edge learns sin(2x) to MSE < 1e-3 (recorded run: 4.82e-06).

## 5. `KANConv1d`: a KAN that slides over days

**Why:** the 60-day sensing window has patterns over a few consecutive days (e.g. three short
nights in a row). A convolution looks at a small window and slides it along time.

```python
def forward(self, x):                      # x: (B, 60, D)
    win = x.unfold(1, self.k, 1)           # (B, 60-k+1, D, k): every window of k consecutive days
    win = win.reshape(B, 60-k+1, D*k)      # flatten features x day offsets
    return self.kan(win)                   # KANLinear(D*k -> C) on every window: (B, 60-k+1, C)
```

- "Valid" padding: no window reaches beyond the last day, so no future day is ever created.
- `test_conv_matches_equation_3` checks the fast version against a slow loop of the paper's Eq. 3.

## 6. KAN-MTLNet (`model.py`): the reproduced model

```
daily series (B,60,22) -> KANConv1d k=3 (22 ch) ┐
                       -> KANConv1d k=5 (21 ch) ├-> average over time -> concat (64) ┐
                       -> KANConv1d k=7 (21 ch) ┘                                     ├-> LayerNorm -> shared KAN -> u (64)
engineered E (B,34) ------------------------------------------------------------------┘
static S (B,10) --> static KAN -> s (8)
[u ; s] -> LayerNorm -> depression KAN head -> logit
                     -> anxiety    KAN head -> logit
```

- **Multi-scale:** windows of 3, 5 and 7 days catch short and longer patterns.
- **Multitask learning:** one shared body, two heads (depression, anxiety). **Why:** the two are
  related, so sharing lets each task learn from the other's examples. **Common mistake:** assuming
  multitask always helps; it was not tested separately here (it is one of the planned ablations).
- Choices the paper does not fix are written in the file docstring (64 channels split 22/21/21,
  G = 5, k = 3, grid [-3, 3], LayerNorms).

## 7. Focal loss and the multitask loss (`model.py`)

**Why:** only about 12–15% of samples are screen-positive. Plain cross-entropy is dominated by the
many easy negatives. **Focal loss** down-weights examples the model already gets right.

```
p_t = p if y = 1 else 1 - p          (probability given to the TRUE class)
FL  = alpha_t * (1 - p_t)^gamma * BCE,     alpha = 0.25, gamma = 2
```

Worked example (positive case, y = 1, α_t = 0.25):

| model probability p | BCE = −ln p | (1 − p)² | focal loss |
|---|---|---|---|
| 0.9 (easy, right) | 0.105 | 0.01 | 0.25 × 0.01 × 0.105 ≈ **0.0003** |
| 0.1 (hard, wrong) | 2.303 | 0.81 | 0.25 × 0.81 × 2.303 ≈ **0.466** |

The hard example counts about 1,800 times more. `multitask_loss` = 1.2 × focal(depression) +
0.8 × focal(anxiety) (paper Eq. 6).

## 8. Training (`train.py`)

| Setting | Value (`train.py`) |
|---|---|
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Batch | 128 |
| Early stopping | on validation **macro AUROC**, patience 10, max 100 epochs |
| Seed | `--seed` (0 reported) |
| Test metrics | AUROC, AUC-PR per task, macro / micro; student-level bootstrap CI (`student_bootstrap`) |

`test_model_overfits_small_batch` checks that the full model can memorise 32 samples (accuracy 1.00).

## 9. Detection result and what it means

From `README.md` (seed 0, 43 test students, 4,826 windows). **The result files (`runs/seed0/`) are
not on this laptop**, so these numbers cannot be re-checked here:

| Task | AUROC [95% CI] | AUC-PR | No-skill AUC-PR |
|---|---|---|---|
| Depression (PHQ-2 ≥ 3) | 0.585 [0.531, 0.640] | 0.181 | 0.149 |
| Anxiety (GAD-2 ≥ 3) | 0.582 [0.524, 0.639] | 0.170 | 0.140 |
| Macro | 0.583 [0.533, 0.636] | 0.176 | – |

- **Meaning:** better than a coin flip (0.5, the CI excludes it) but weak. AUC-PR is only slightly
  above its no-skill level (= the share of positives).
- The paper reports 0.731 on the All of Us cohort; README explains why the numbers are not
  comparable (Fitbit heart rate / HRV vs phone data, clinical labels vs screening scores, ~40,000
  people vs 214 students).
- Early stopping chose epoch 6; validation AUROC fell afterwards while training loss kept falling:
  the model **overfits** the 149 training students quickly.
- **Link to later results:** phone sensing carried little signal here, and this was confirmed in the
  forecasting work (version A ≈ mean-score baseline; history-only control).

## 10. What is and is not shown

- **Shown:** the reproduction runs and passes unit tests; on CES phone data it detects current screen
  status only weakly.
- **Not shown:** that KANs are better or worse than MLPs here (no MLP comparison was run); that
  the weak result is due to the model rather than the data.
