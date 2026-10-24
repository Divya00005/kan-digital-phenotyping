# 03 · ODEs, RK4, KAN-ODE (variants A and C) and how they were trained

> Code: `ode.py` (`rk4_trajectory`), `forecast_model.py` (`Encoder`, `KANDynamics`, `Head`,
> `KANODEForecaster`, `NoODEForecaster`, `forecast_loss`), `odernn_model.py`
> (`ODERNNForecaster`, `GRUDtForecaster`), `train_forecast.py`, `runs/pipeline_456.py` (git-ignored
> driver for variant A steps 4–6). Tests: `test_ode.py`, `test_odernn.py`.

## 1. What a differential equation is

**Why:** symptoms and behaviour change *continuously* over time, and our surveys are irregular
(every ~5 days, with gaps). A model that describes **how a state changes per unit of time** can be
asked "where will it be in 7 days? in 15 days?" for any number of days.

An **ordinary differential equation (ODE)** says how fast something changes:

```
dz/dt = f(z)          "the rate of change of z depends on the current z"
```

**Everyday example:** a cup of tea cooling. The hotter it is compared with the room, the faster it
cools: dT/dt = −k (T − T_room). Knowing today's temperature and this rule, you can predict the
temperature at any later time by "stepping forward".

**Hidden state:** in our models z is not a symptom score but a 16-number summary of the student
(a **latent / hidden state**) produced by the encoder. The ODE moves that summary forward in time,
and a head turns it into scores. Nobody tells the model what the 16 numbers mean; it learns them.

## 2. Solving an ODE step by step: Euler and RK4

We cannot usually solve dz/dt = f(z) on paper, so we take small time steps of size dt.

**Euler (simplest):** `z_next = z + dt * f(z)` — "go straight in the current direction".
**RK4 (`ode.py`)** looks at the slope four times per step and averages:

```python
k1 = f(z)
k2 = f(z + 0.5*dt*k1)
k3 = f(z + 0.5*dt*k2)
k4 = f(z + dt*k3)
z  = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

### Worked example by hand
dz/dt = −z, start z = 1, one step of dt = 0.5. Exact answer: e^(−0.5) = **0.60653**.

| Method | Calculation | Result | Error |
|---|---|---|---|
| Euler | 1 + 0.5 × (−1) | 0.50000 | 0.107 |
| RK4 | k1 = −1; k2 = −(1 − 0.25) = −0.75; k3 = −(1 − 0.1875) = −0.8125; k4 = −(1 − 0.40625) = −0.59375; z = 1 + (0.5/6)(−1 − 1.5 − 1.625 − 0.59375) | 0.60677 | 0.00024 |

RK4 is ~400 times more accurate here for four function evaluations.

**In this project:** time unit = weeks; one step = **1 day** (`DAY = 1/7` in `forecast_model.py`);
`MAX_DAYS = 59` (8 weeks + 3 days tolerance). Because every target offset is a whole number of
days, every target lands exactly on a step — no interpolation needed.

**Tests (`test_ode.py`):** RK4 vs the exact solution of dz/dt = −z after 8 weeks (recorded error
2.1e-08); RK4 vs the adaptive `torchdiffeq` dopri5 solver on a random KAN dynamics (recorded max
difference 1.7e-06).

## 3. Backpropagation through the solver

The solver is just additions and multiplications, so PyTorch's autograd can send gradients back
through all 59 steps to the dynamics and the encoder ("discretise-then-optimise"; no adjoint method).
`test_gradients_reach_dynamics_and_encoder` checks that both receive non-zero gradients.

**Cost:** 59 steps × 4 evaluations = 236 sequential KAN calls per forward pass. Measured: ~2.3–2.6 s
per batch, ~1.5–2.5 min per epoch at batch 512, vs ~7 s per epoch for the no-ODE model. This is why
batch 512 was chosen (same for both models).

## 4. The forecasting models (`forecast_model.py`)

### 4.1 Shared encoder (`Encoder`)
The KAN-MTLNet body from file 02 (multi-scale `KANConv1d`, engineered features, optional history
`h`, static KAN) ending in `to_latent`: **z(t0), 16 numbers**.

### 4.2 Variant A: `KANODEForecaster` (ODE after the encoder)

```
inputs -> Encoder -> z(t0) --RK4, 1-day steps, dz/dt = KAN(z)--> z(t0+1d) ... z(t0+59d)
                                         pick the steps at the real offsets (7, 15, 26, 55 days)
                                         -> Head (LayerNorm + KANLinear 16->4)
                                         -> [dep score, anx score, dep logit, anx logit] per horizon
```

```python
class KANDynamics:     # f(z) = dz/dt
    l1 = KANLinear(16, 32, grid_range=(-1, 1)); l2 = KANLinear(32, 16, grid_range=(-1, 1))
    forward(z) = l2(tanh(l1(tanh(z))))      # tanh keeps inputs inside the [-1, 1] grid
```

### 4.3 The no-ODE control: `NoODEForecaster`
**Why a control:** if the KAN-ODE did better than persistence, we would not know whether the
**ODE** or just the **extra KAN layers** helped. The control has the **same encoder and the same
size KAN** (`KANDynamics(16, 32, extra_in=1)`) but applies it **once** to [z(t0), τ], where
τ = offset_days / 7 / 8 tells it the horizon. No integration.

Parameter counts (from training logs): version B 144,336 (ODE) vs 144,624 (no-ODE); version A
140,868 vs 141,156 — matched within 0.2%. So any difference comes from the integration itself.

`test_offset_zero_is_initial_state` checks that offset 0 returns exactly the head applied to z(t0)
(time indexing is right).

### 4.4 Variant C: ODE **inside** the encoder (`odernn_model.py`)
**Why tried:** variant A showed no benefit. Rubanova et al. (2019) proposed an **ODE-RNN** for
irregularly sampled series: between observations the hidden state flows by an ODE; at an
observation a GRU updates it. CES is irregular in exactly two ways: missing sensing days and
surveys every ~5 days.

```
ODE-RNN (ODERNNForecaster), hidden h (32):
 for day k = 0 .. 60 (k = 60 is t0):
     if k > 0:  h <- RK4 step of dh/dt = KAN(h) over 1 day
     if day k has valid sensing:   h <- GRU_x(x_k, h)
     if day k has a survey:        h <- GRU_s(scores_k, h)
 GRU-dt control (GRUDtForecaster): same updates, h frozen between them, and the gap since the
 last update (in weeks) is an extra GRU input; the same KAN block applied once at the end.
```

Both then combine the final h with `E`, `H`, `S` and use the **no-ODE head** of variant A.
Parameters: 90,428 (ODE-RNN) vs 90,620 (GRU-Δt), +0.21% (`test_odernn.py`).

## 5. The loss (`forecast_model.forecast_loss`)

```
loss = MSE(predicted scores, true scores) + 10 * [1.2 * focal(dep screen) + 0.8 * focal(anx screen)]
```
computed only over (cutoff, horizon) pairs with a target (`mask`). The score part teaches the
number; the focal part teaches "≥ 3 or not" and handles the 12–15% positive rate.

## 6. Training procedure (`train_forecast.py → train_one`)

| Setting | Value | Why |
|---|---|---|
| Optimiser | AdamW, weight decay 1e-4 | as in the detection model |
| Learning rate | chosen on validation (below) | tuning only on validation |
| Batch | 512 | ODE ~4× faster per epoch at the same GPU cost; same for all models |
| Gradient clipping | norm 1.0 | protects against ODE blow-ups |
| Early stopping | validation loss, patience 10, max 100 epochs | models overfit after a few epochs |
| Seeds | 0–4, predictions averaged | reduces run-to-run noise |
| NaN guard | stop with an error | never silently continue |

**Settings chosen on validation (variant A, `runs/tune/selection.json`):** four settings were listed
*before* running: C0 lr 1e-3; C1 lr 3e-4; C2 residual (predict change from today) lr 1e-3; C3
residual lr 3e-4. Rule: lowest mean validation MAE over both models; ties within 0.005 → simpler.

| Setting | Version A val MAE | Version B val MAE |
|---|---|---|
| C0 | 1.098 | 0.7214 → **chosen (tie with C3, simpler)** |
| C1 | **1.0448 → chosen** | 0.7378 |
| C2 | – | 0.7258 |
| C3 | – | 0.7185 |

**Variant C (`runs/variantC/tune/selection.json`):** lr 1e-3 (mean val MAE 0.7138) vs 3e-4 (0.7317)
→ 1e-3.

**Seed spread:** each seed alone differs by about ±0.01 MAE (SD 0.004–0.013 in
`runs/final/eval_test/rq1_table.md`), much smaller than the differences we care about vs
persistence, but similar to the tiny ODE vs no-ODE differences.

**Runs:** variant A final = 2 models × 2 versions × 5 seeds = 20 runs (`runs/final/`), trained by
`runs/pipeline_456.py`; variant C = 2 models × 5 seeds (`runs/variantC/final/`).

## 7. What the tests check

| Test file | Checks |
|---|---|
| `test_ode.py` | RK4 exactness; agreement with torchdiffeq; offset-0 indexing; gradients flow; both forecasters can memorise a small batch (recorded MAE 0.11 ODE, 0.29 no-ODE) |
| `test_odernn.py` (9 tests) | parameter match (+0.21%); values on invalid days / non-survey days are ignored; no observations → pure ODE solution; zero dynamics → plain GRU; GRU-Δt gets exact gaps (10, 15, 35 days); both react to *when* a survey happened; batch independence; gradients reach all parts; small-batch memorisation (MAE 0.17 / 0.24) |

## 8. Common mistakes to avoid when explaining

- "The ODE models symptoms directly" — no, it moves a learned hidden state.
- "RK4 is approximate so results are noisy" — the solver error is ~1e-6; noise comes from data.
- "The ODE has more parameters, so a fair comparison is impossible" — the control is parameter-matched.
- "ODEs are needed because surveys are irregular" — that was the hypothesis; the results (file 04)
  show the no-ODE control handled irregular offsets just as well (it receives the offset as input).
