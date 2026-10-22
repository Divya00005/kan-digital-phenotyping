# kan-digital-phenotyping

Research code for *Longitudinal, Causally-Informed Multitask Digital Phenotyping with
Uncertainty-Aware KAN Models* (B.Tech HCI + AI project). This is a research tool, not a
clinical diagnostic system.

**Stages so far:**
1. Re-implementation of **Interpretable-MTLNet** (Ali, Ni & Kundur, NeurIPS 2025 workshop)
   aligned to the College Experience Study (CES) dataset. Detection setting, same as the paper:
   predict current depression / anxiety screen status from the 60 days of phone sensing before a survey.
2. **RQ1, forecasting with KAN-ODE** (variant A): future symptoms 1, 2, 4 and 8 weeks ahead.
   Test-set results below (negative for the ODE).

Later stages (factor analysis, conformal prediction) build on these models.

## Data

The CES dataset is **not** included in this repository. Download it from Kaggle:
<https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset>.
The `.gitignore` blocks data files (`*.csv`, `*.npz`, `*.zip`) so they are never committed.

## Files

| File | What it does |
|---|---|
| `data_prep.py` | Builds the aligned dataset (`ces_kan_dataset.npz`) from the CES CSVs |
| `kan.py` | `KANLinear` (paper Eq. 1-2) and `KANConv1d` (Eq. 3) |
| `model.py` | Full KAN-MTLNet + weighted focal loss (Eq. 4-7) |
| `train.py` | Training recipe of Appendix B + test metrics + student-level bootstrap CI |
| `test_kan.py` | Unit tests for the KAN pieces |

## How to run

```bash
pip install -r requirements.txt
python data_prep.py --ema EMA/general_ema.csv --sensing Sensing/sensing.csv \
                    --demo Demographics/demographics.csv --out ces_kan_dataset.npz
python test_kan.py
python train.py --data ces_kan_dataset.npz --seed 0 --out runs/seed0
```

## How the paper is aligned to CES

| Paper (All of Us) | This implementation (CES) | Why |
|---|---|---|
| ~40,000 participants, one 60-day window each | 214 students, one 60-day window **per PHQ-4 survey** (27,925 windows) | CES has repeated surveys; each survey is a labelled time point |
| Window = 60 days of Fitbit data | Window = days *t-60 ... t-1* before survey day *t* (survey day excluded) | No information from the survey day leaks in |
| Labels: clinically validated depression / anxiety | `PHQ-2 >= 3` (items 3+4) and `GAD-2 >= 3` (items 1+2) | Standard screen-positive cutoffs. **Screening, not diagnosis** |
| Daily steps, HR, HRV, sleep stages | 21 cross-platform phone features: activity, mobility, places, phone use, sleep | Audit: Android-only / iOS-only features are zero-filled on the other platform |
| Missing: train median + indicators | Day invalid if activity or location quality < 12 h -> features missing; train-median imputation; daily valid-mask channel | Audit: missingness rises from 6% (2017) to 42% (2022) |
| Engineered features (variability, regularity) | Mean and coefficient of variation over 3/7/14 days, weekend-weekday difference, valid-day fractions (34 features) | Same idea, our features |
| Static covariates | Gender, race group, iPhone share | Only demographics released in CES |
| Subject-level 70/10/20 split | Student-level 70/10/20 split (149/22/43) | Same |
| z-score with training statistics | Same (fit on training students only) | Same |

### Choices the paper does not specify (documented, not hidden)

- Grid size G = 5, spline order k = 3 (KAN paper defaults), grid range [-3, 3]
- 64 channels **in total** across the three kernel sizes (22/21/21), following Appendix B
- LayerNorm before the shared and head KAN layers to keep hidden values inside the spline grid
- Survey kept only if at least 7 of the 14 days before it are valid
- Sleep duration outside 1-16 h treated as missing

### Training recipe (paper Appendix B)

AdamW, lr 1e-3, weight decay 1e-4, batch 128, focal loss (alpha 0.25, gamma 2),
task weights 1.2 (depression) / 0.8 (anxiety), early stopping on validation macro-AUROC
with patience 10, maximum 100 epochs.

### Evaluation

Per-task AUROC and AUC-PR, macro and micro averages (paper Appendix A), plus
95% confidence intervals from a **student-level bootstrap**, because one student contributes
many overlapping windows. AUC-PR should be read against its no-skill value (the prevalence).

## First result (single seed, preliminary)

Seed 0, 43 held-out test students (4,826 windows), 95% CI from a student-level bootstrap:

| Task | AUROC [95% CI] | AUC-PR | No-skill AUC-PR |
|---|---|---|---|
| Depression (PHQ-2 >= 3) | 0.585 [0.531, 0.640] | 0.181 | 0.149 |
| Anxiety (GAD-2 >= 3) | 0.582 [0.524, 0.639] | 0.170 | 0.140 |
| Macro | 0.583 [0.533, 0.636] | 0.176 | - |

Early stopping chose epoch 6; validation AUROC fell afterwards while training loss kept
decreasing, so the model overfits the 149 training students quickly. These numbers are
not comparable to the paper's 0.731 (All of Us, Fitbit heart rate/HRV, clinical labels,
~40,000 people). Five-seed results will replace this table.

Dataset citation: Nepal et al. (2024), *Capturing the College Experience*, Proc. ACM IMWUT 8(1), Article 38.

## RQ1: forecasting future symptoms with KAN-ODE

**Question:** can a KAN-ODE predict future depression / anxiety scores (1, 2, 4, 8 weeks ahead)
better than persistence and than the same KAN without an ODE?

**Answer on the test set (variant A of the KAN-ODE): no.** The ODE did not improve on the
parameter-matched no-ODE control (pre-registered decision rule: NO for both input versions),
and no KAN model beat persistence ("future score = today's score") on score error.
With symptom history as input, both KAN models ranked screen-positive and new-onset risk
better than persistence, but the ODE added nothing to that.

| File | What it does |
|---|---|
| `forecast_prep.py` | Forecasting dataset: cutoff = survey day t0, 60 days of sensing before t0, targets = PHQ-2 / GAD-2 at t0 + 1, 2, 4, 8 weeks (closest survey within +/- 3 days, exact day offset kept). Also computes persistence and mean baselines |
| `ode.py` | Fixed-step RK4 solver (1-day steps), backpropagation through the solver |
| `forecast_model.py` | `KANODEForecaster`: encoder -> z(t0) -> dz/dt = KAN(z) -> heads at the target day. `NoODEForecaster`: same encoder and KAN sizes, horizon given as input, no integration (control) |
| `test_ode.py` | Tests: RK4 vs exact solution and vs `torchdiffeq`, time indexing, gradients, small-batch fitting |
| `train_forecast.py` | Trains one model (`--model {ode,noode} --version {A,B} --seed N`) or all of them (`--run_all --seeds ...`) on the GPU; uses training and validation students only; saves the best-epoch model and validation predictions |
| `evaluate_forecast.py` | RQ1 table: mean score, persistence, no-ODE and KAN-ODE per horizon; student-level bootstrap CIs, paired CIs for differences, the decision rule. Validation by default; `--split test --final` works once (lock file) |

```bash
python forecast_prep.py --ema EMA/general_ema.csv --sensing Sensing/sensing.csv \
    --demo Demographics/demographics.csv --split_from ces_kan_dataset.npz --out ces_forecast.npz
python test_ode.py
# final settings (chosen on validation, see below)
python train_forecast.py --run_all --versions A --seeds 0 1 2 3 4 --lr 3e-4 --out_root runs/final
python train_forecast.py --run_all --versions B --seeds 0 1 2 3 4 --out_root runs/final
python evaluate_forecast.py --runs runs/final --seeds 0 1 2 3 4                         # validation
python evaluate_forecast.py --runs runs/final --split test --final --seeds 0 1 2 3 4   # once
```

Input versions: **A** = sensing only; **B** = sensing + symptom history known at t0
(today's scores, 28-day mean, days since previous survey, number of recent surveys).

### Persistence baseline (test students; future score = today's score)

| Horizon | Depression MAE (mean / persistence) | Persistence AUROC | Onset rate | Onset AUC-PR |
|---|---|---|---|---|
| 1 week | 1.184 / 0.600 | 0.894 | 5.6% | 0.127 |
| 2 weeks | 1.189 / 0.686 | 0.865 | 6.7% | 0.149 |
| 4 weeks | 1.210 / 0.773 | 0.847 | 7.4% | 0.148 |
| 8 weeks | 1.221 / 0.874 | 0.810 | 9.2% | 0.173 |

Onset = currently below 3, at or above 3 at the horizon.

### Training and evaluation (fixed before the test set was used)

- **Models:** KAN-ODE (variant A, ~141-144k parameters) and the no-ODE control (parameter-matched,
  within 0.2%). Same encoder, same KAN sizes; the only difference is integrating dz/dt = KAN(z) vs
  a single KAN step with the horizon as input.
- **Training:** loss = score MSE + 10 x focal loss on screen status (task weights 1.2 / 0.8),
  AdamW (weight decay 1e-4), batch 512, gradient-norm clipping 1.0, early stopping on validation
  loss (patience 10, max 100 epochs). Batch 512 instead of 128 because the ODE is ~4x faster per
  epoch at the same GPU cost; both models use identical settings.
- **Primary metric:** MAE of the future score averaged over depression and anxiety (model scores
  clipped to 0-6). Secondary: per-task MAE, AUROC / AUC-PR for screen-positive (>= 3), onset
  AUROC / AUC-PR among students currently below 3.
- **Seeds:** 5 seeds (0-4) per model; predictions averaged over seeds; the spread over seeds is
  also reported (SD of the primary metric about 0.01).
- **CIs:** 1,000 student-level bootstrap resamples; all methods are scored on the same resamples,
  so differences are paired.
- **Decision rule (pre-registered):** "the ODE helps" only if the paired 95% CI of
  MAE(KAN-ODE) - MAE(no-ODE) is entirely below zero at >= 2 of the 4 horizons.
- **Test set used once** (`runs/TEST_SET_USED.json`, 2026-10-22); 43 test students, 4,733 cutoffs.

**Settings chosen on validation (seed 0, 22 students).** Four settings, listed before any were
run; the setting with the lowest validation MAE (mean over horizons and over both models, so both
models share it) was kept, with ties within 0.005 going to the simpler one:

| Setting | Version A | Version B |
|---|---|---|
| C0: lr 1e-3 | 1.098 | 0.721 **(chosen: tie with C3)** |
| C1: lr 3e-4 | **1.045 (chosen)** | 0.738 |
| C2: residual (predict the change from today's score), lr 1e-3 | - | 0.726 |
| C3: residual, lr 3e-4 | - | 0.719 |

### Results (test students, evaluated once)

Primary metric: MAE averaged over depression and anxiety, lower is better, [95% CI].

**Version B: sensing + symptom history known at t0**

| Horizon | Mean score | Persistence | KAN no-ODE | KAN-ODE | KAN-ODE - no-ODE |
|---|---|---|---|---|---|
| 1 week | 1.152 | **0.633** [0.535, 0.725] | 0.754 [0.636, 0.906] | 0.739 [0.624, 0.880] | -0.015 [-0.030, -0.005] |
| 2 weeks | 1.159 | **0.715** [0.601, 0.820] | 0.788 [0.666, 0.937] | 0.785 [0.667, 0.926] | -0.004 [-0.017, 0.004] |
| 4 weeks | 1.174 | **0.805** [0.680, 0.920] | 0.855 [0.719, 1.017] | 0.864 [0.735, 1.017] | +0.009 [-0.005, 0.019] |
| 8 weeks | 1.195 | **0.917** [0.754, 1.060] | 0.937 [0.785, 1.105] | 0.946 [0.801, 1.107] | +0.009 [-0.006, 0.023] |

Decision rule: ODE better at 1 week only -> **ODE helps: NO**. Both models are worse than
persistence at 1 week (CI of the difference excludes zero) and not different from it at 2-8 weeks.

**Version A: sensing only**

| Horizon | Mean score | Persistence | KAN no-ODE | KAN-ODE | KAN-ODE - no-ODE |
|---|---|---|---|---|---|
| 1 week | 1.152 | **0.633** | 1.140 | 1.138 | -0.002 [-0.014, 0.012] |
| 2 weeks | 1.159 | **0.715** | 1.143 | 1.142 | -0.001 [-0.014, 0.014] |
| 4 weeks | 1.174 | **0.805** | 1.156 | 1.157 | +0.001 [-0.012, 0.017] |
| 8 weeks | 1.195 | **0.917** | 1.181 | 1.178 | -0.002 [-0.016, 0.016] |

Decision rule: **ODE helps: NO**. Neither model differs from predicting the training mean
(screen-positive AUROC 0.53-0.62): phone sensing alone carries almost no forecasting signal here.

**Secondary metrics, version B** (persistence / no-ODE / KAN-ODE):

| Horizon | Depression AUROC | Anxiety AUROC | Depression onset AUC-PR | Anxiety onset AUC-PR |
|---|---|---|---|---|
| 1 week | 0.894 / 0.906 / 0.907 | 0.864 / 0.882 / 0.885 | 0.127 / 0.226 / 0.226 | 0.117 / 0.245 / 0.252 |
| 2 weeks | 0.865 / 0.889 / 0.891 | 0.849 / 0.877 / 0.879 | 0.149 / 0.273 / 0.270 | 0.123 / 0.258 / 0.258 |
| 4 weeks | 0.847 / 0.872 / 0.870 | 0.814 / 0.853 / 0.853 | 0.148 / 0.257 / 0.257 | 0.119 / 0.217 / 0.218 |
| 8 weeks | 0.810 / 0.830 / 0.830 | 0.742 / 0.792 / 0.794 | 0.173 / 0.268 / 0.270 | 0.121 / 0.204 / 0.207 |

The KAN-ODE beats persistence on screen-positive AUROC at 6 of 8 horizon-task pairs and on onset
AUC-PR at all 8 (paired CIs exclude zero); the no-ODE model performs the same, so the gain is not
due to the ODE. Full tables with CIs: `runs/final/eval_test/rq1_table.md` (not in git).

### Limitations

- Version B has no "symptom history only" model, so the ranking gain cannot be attributed to
  phone sensing rather than to the history features.
- Only 22 validation students: model selection was noisy (validation CIs about +/- 0.15 MAE).
- Scores are screening scores (PHQ-2 / GAD-2), not diagnoses; surveys are irregular (about every 5 days).
- Only KAN-ODE variant A was tested. The agreed plan names variant C (KAN ODE-RNN encoder,
  Rubanova et al. 2019) as the next step when variant A shows no ODE benefit.
