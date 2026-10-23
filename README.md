# kan-digital-phenotyping

Research code for *Longitudinal, Causally-Informed Multitask Digital Phenotyping with
Uncertainty-Aware KAN Models* (B.Tech HCI + AI project). This is a research tool, not a
clinical diagnostic system.

**Stages so far:**
1. Re-implementation of **Interpretable-MTLNet** (Ali, Ni & Kundur, NeurIPS 2025 workshop)
   aligned to the College Experience Study (CES) dataset. Detection setting, same as the paper:
   predict current depression / anxiety screen status from the 60 days of phone sensing before a survey.
2. **RQ1, forecasting with KAN-ODE**: future symptoms 1, 2, 4 and 8 weeks ahead. Two ODE designs
   tested (variant A: ODE after the encoder; variant C: ODE-RNN encoder). Test-set results below:
   negative for the ODE in both.
3. **History-only control** (cross-validation, test set not used): the models' ranking gain over
   persistence comes from the symptom-history features, not from phone sensing.
4. **RQ3, conformal prediction sets** (nested cross-validation, test set not used): student-level
   conformal sets reach their 90% / 80% coverage target for every model, overall and within
   female, male, White and Asian students; ignoring that surveys come from students does not.

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
| `forecast_prep.py` | Forecasting dataset: cutoff = survey day t0, 60 days of sensing before t0, targets = PHQ-2 / GAD-2 at t0 + 1, 2, 4, 8 weeks (closest survey within +/- 3 days, exact day offset kept). Also computes persistence and mean baselines, and the past-survey sequence used by variant C |
| `ode.py` | Fixed-step RK4 solver (1-day steps), backpropagation through the solver |
| `forecast_model.py` | `KANODEForecaster`: encoder -> z(t0) -> dz/dt = KAN(z) -> heads at the target day. `NoODEForecaster`: same encoder and KAN sizes, horizon given as input, no integration (control) |
| `test_ode.py` | Tests: RK4 vs exact solution and vs `torchdiffeq`, time indexing, gradients, small-batch fitting |
| `train_forecast.py` | Trains one model (`--model {ode,noode} --version {A,B} --seed N`) or all of them (`--run_all --seeds ...`) on the GPU; uses training and validation students only; saves the best-epoch model and validation predictions |
| `evaluate_forecast.py` | RQ1 table: mean score, persistence, no-ODE and KAN-ODE per horizon; student-level bootstrap CIs, paired CIs for differences, the decision rule. Validation by default; `--split test --final` works once (lock file). `--control` / `--treatment` / `--ref_runs` for variant C |
| `odernn_model.py` | Variant C: `ODERNNForecaster` (KAN ODE-RNN encoder) and the `GRUDtForecaster` control |
| `test_odernn.py` | Tests: missing days ignored, zero dynamics = plain GRU, empty walk = pure ODE, exact time gaps, batch independence, gradients, small-batch fitting |
| `baseline_history.py` | History-only control: linear / logistic regression on history + static features, per CV fold |
| `evaluate_history_control.py` | History-only control: pooled out-of-fold table, paired bootstrap, the decision rule |
| `test_history_control.py` | Tests: folds complete and student-disjoint, no test rows, version H ignores sensing, same model size, baseline |
| `conformal.py` | RQ3 conformal methods (numpy only, works with any model): split conformal, student-level repeated subsampling, pooled CDF, CV+, CQR score, per-group calibration, joint coverage |
| `quantile_model.py` | RQ3: KAN no-ODE with 5 / 10 / 90 / 95% quantile outputs (pinball loss) for CQR |
| `rq3_conformal.py` | RQ3 evaluation: prediction sets for every model and method, coverage overall and by group, bootstrap CIs, flags, verdict |
| `test_conformal.py` | Tests of the conformal methods on synthetic data |
| `test_rq3_training.py` | Tests: nested folds disjoint and complete, no test rows, version Hn ignores gender / race, pinball loss, quantile models |

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
- Variant C (below) was designed after these results were known.

## RQ1 variant C: ODE inside the encoder (KAN ODE-RNN)

**Question:** variant A put the ODE after the encoder. Does an ODE **inside** the encoder help,
reading the 61 days before t0 as an irregular time series (missing sensing days, surveys about
every 5 days)? Pre-registered in `CLAUDE.md` and committed (44538b5) before any variant C code.

**Answer on the test set: no.** KAN ODE-RNN and its GRU-Δt control are practically identical
(differences at most 0.003 MAE); neither beats persistence, and both are slightly worse than the
variant A models in point estimates (variant A is a reference row only, no paired test).

- **KAN ODE-RNN** (Rubanova et al. 2019): hidden state h (32) walks day by day; between days
  dh/dt = KAN(h) (one RK4 step per day); a GRU update on each valid sensing day and on each survey
  day; no update on missing days.
- **GRU-Δt control:** the same GRU updates on the same days, no ODE; the time since the previous
  update is an extra input; the same-sized KAN block applied once at the end. Parameter counts
  90,428 vs 90,620 (+0.2%).
- Both: input version B plus the past-survey sequence (PHQ-2 / GAD-2 on each survey day in
  t0-60 .. t0); the no-ODE head of variant A. Same training recipe as variant A.
- **Learning rate** chosen on validation (seed 0) from {1e-3, 3e-4}: 1e-3 (mean val MAE 0.714 vs 0.732).
- Seeds 0-4, predictions averaged. Same metrics, bootstrap and decision rule as variant A.
- **Second use of the test set** (2026-10-22, lock `runs/TEST_SET_USED_variantC.json`).

```bash
python test_odernn.py
python train_forecast.py --run_all --models grudt odernn --versions B --seeds 0 1 2 3 4 \
    --lr 1e-3 --out_root runs/variantC/final
python evaluate_forecast.py --control grudt --treatment odernn --versions B \
    --runs runs/variantC/final --ref_runs runs/final --ref_models noode ode \
    --split test --final --seeds 0 1 2 3 4 --lock runs/TEST_SET_USED_variantC.json   # once
```

**Results (test students; MAE averaged over depression and anxiety, [95% CI])**

| Horizon | Persistence | Variant A KAN-ODE (ref) | GRU-Δt | KAN ODE-RNN | ODE-RNN - GRU-Δt |
|---|---|---|---|---|---|
| 1 week | **0.633** | 0.739 | 0.763 [0.649, 0.910] | 0.766 [0.648, 0.922] | +0.003 [-0.007, 0.015] |
| 2 weeks | **0.715** | 0.785 | 0.805 [0.687, 0.947] | 0.805 [0.684, 0.957] | +0.000 [-0.009, 0.012] |
| 4 weeks | **0.805** | 0.864 | 0.880 [0.750, 1.028] | 0.880 [0.745, 1.037] | -0.000 [-0.010, 0.012] |
| 8 weeks | **0.917** | 0.946 | 0.962 [0.816, 1.115] | 0.961 [0.809, 1.122] | -0.001 [-0.011, 0.012] |

Decision rule: better at no horizon -> **ODE-RNN helps: NO**. Both models are worse than
persistence at 1 week (CI excludes zero) and not different from it at 2-8 weeks.

**Secondary metrics:** both models beat persistence on screen-positive AUROC (7 of 8 horizon-task
pairs) and onset AUC-PR (8 of 8), at the same level as variant A (e.g. 1 week onset AUC-PR about
0.23-0.25 vs persistence 0.12). Of 16 secondary ODE-RNN vs GRU-Δt comparisons, 2 have a CI
excluding zero, both slightly in favour of the control (anxiety onset AUC-PR at 1 and 4 weeks,
-0.010); with this many comparisons that is consistent with chance.

**What this means for RQ1:** in CES, with these designs, an ODE does not improve forecasts of
PHQ-2 / GAD-2, whether it acts after the encoder (variant A) or inside it (variant C). Reading
the survey history as a time series did not beat the six summary history features either.

**Limitations (in addition to those of variant A):** the test set had already been used once
for variant A, and variant C was designed knowing that result (reported here, not hidden);
validation selection used only 22 students; variant C was tested for input version B only.

## History-only control: does phone sensing add anything?

**Question:** the version B models (sensing + symptom history) rank screen-positive and onset risk
better than persistence. Does that gain come from **phone sensing** or only from the
**symptom-history** features? Predictive question only, no causal claim. Pre-registered in
`CLAUDE.md` and committed (e9b5230) before any code.

**Answer: no evidence that sensing adds ranking information.** The gain over persistence comes
from the history features: the same model without sensing ranks as well or slightly better, and
a plain linear / logistic regression on history does as well as either KAN model.

- **Design:** 5-fold cross-validation by student on the 170 train + validation students
  (22,699 cutoffs); **the test set was not used**. Each student is predicted once by models that
  never saw them (34 held out per fold; 14 more held out for early stopping).
- **Version B:** variant A no-ODE model, sensing + history + static inputs.
  **Version H:** the same model (same size, 144,624 parameters) with the sensing inputs set to zero.
  **Linear / logistic baseline:** on history + static features only.
- Variant A version-B settings, no tuning; 5 seeds per fold, predictions averaged over seeds,
  folds pooled; 1,000-resample paired student bootstrap.
- **Decision rule (pre-registered):** sensing adds ranking information only if the CI of
  AUROC(B) - AUROC(H) is entirely above zero at >= 2 of 4 horizons.

```bash
python test_history_control.py
python baseline_history.py --out runs/historyctl/baseline
python train_forecast.py --run_all --models noode --versions B H --folds 0 1 2 3 4 \
    --seeds 0 1 2 3 4 --out_root runs/historyctl
python evaluate_history_control.py
```

**Results (170 students, out-of-fold).** Screen-positive AUROC averaged over depression and
anxiety, higher is better, [95% CI]:

| Horizon | Persistence | Linear / logistic on history | KAN version H (history only) | KAN version B (sensing + history) | B - H |
|---|---|---|---|---|---|
| 1 week | 0.882 | **0.910** | 0.907 | 0.902 | -0.006 [-0.010, -0.002] |
| 2 weeks | 0.852 | **0.886** | 0.885 | 0.881 | -0.004 [-0.008, -0.000] |
| 4 weeks | 0.832 | 0.867 | 0.867 | 0.865 | -0.002 [-0.007, 0.004] |
| 8 weeks | 0.801 | 0.840 | **0.842** | 0.840 | -0.002 [-0.007, 0.004] |

Decision rule: B better at no horizon (B is **worse** at 1 and 2 weeks) -> **sensing adds ranking
information: NO**.

- **All three history-based models beat persistence** on AUROC at every horizon (+0.02 to +0.04,
  all CIs above zero) and roughly double onset AUC-PR (about 0.25-0.28 vs 0.14-0.15).
- **KAN vs linear:** version H and the linear / logistic baseline are indistinguishable
  (AUROC differences <= 0.003, CIs include zero). Version B is slightly worse than the linear
  baseline at 1-2 weeks.
- **Score error (MAE):** no model beats persistence (0.559 / 0.644 / 0.711 / 0.781 at
  1 / 2 / 4 / 8 weeks); among the models, version B is the worst at every horizon
  (B - H = +0.014 to +0.043, all CIs above zero), and the linear baseline the best at 1-2 weeks.
- **Training:** version B reached its best validation loss sooner (median epoch 4 vs 9 for H),
  consistent with the sensing inputs mainly adding room to overfit.
- Spread: SD over seeds <= 0.004 AUROC; over folds about 0.02.
- The "mean score" row of the full table is a no-skill reference: its pooled AUROC (about 0.45)
  only reflects small differences between the folds' training means.

Full tables (onset AUC-PR, AUC-PR, MAE, all paired differences): `runs/historyctl/eval/table.md`
(not in git).

**What this means:** in CES, these 21 daily phone-sensing features did not add predictive
information about future PHQ-2 / GAD-2 screen status beyond the student's own recent survey answers,
for either a KAN or a simple model. Combined with version A (sensing only = no better than the mean)
and the detection model (AUROC 0.58), sensing carries at most a weak signal here.

**Limitations:** inputs were normalised with the original 149 training students (some held-out
students' inputs contributed to those means / SDs; inputs only, no labels; same for B and H);
only the no-ODE KAN and one simple baseline were tried; one dataset; associational only.

## RQ3: conformal prediction sets, overall and by gender / race

**Question:** can conformal prediction give reliable prediction sets for future PHQ-2 (depression)
and GAD-2 (anxiety) scores at 1, 2, 4 and 8 weeks, overall and within gender and race groups?
Pre-registered in `CLAUDE.md` (commit d5faf4e) with a dated amendment (251e369), both before any
RQ3 training.

**Answer: yes, if the calibration treats students (not surveys) as the unit.** With student-level
repeated subsampling (method B), every model reached its 90% and 80% targets overall, and within
female, male, White and Asian students under per-group calibration: all 8 models pass the
pre-registered "reliable" check. The sets are wide (about 3 of the 7 possible scores at 90%), and
under pooled calibration anxiety coverage is a few points lower for female than for male students.

**Design**
- **Students:** nested student-level cross-validation on the 170 train + validation students; **the
  test set was not used.** Each outer fold (34 students) is evaluated by models and calibration data
  that never saw those students. For the main method, the other 136 students are split into 68
  model students (training) and 68 calibration students; for CV+, into 4 inner folds.
- **Models:** persistence, linear model on history, KAN no-ODE H (history) and B (sensing +
  history), KAN no-ODE Hn (H without gender / race inputs, sensitivity check), **KAN-ODE H**, and
  two quantile models for CQR (linear quantile regression, quantile KAN H). One seed.
- **Methods** (`conformal.py`):

| Method | Paper | Role | Guarantee |
|---|---|---|---|
| **B: one random survey per calibration student, 200 subsamples, averaged p-values** | Dunn et al. 2023 | **main** | >= 1 - 2 alpha for a random survey of a new student (in practice about 1 - alpha) |
| B1: single subsample | Dunn et al. 2023 | reference | >= 1 - alpha (exact) |
| A: survey-level split conformal | Lei et al. 2018 | reference | none for a new student (surveys of one student are not exchangeable) |
| Pooled CDF (each student weighted equally) | Dunn et al. 2023 | comparison | asymptotic only |
| B + CV+ over 4 inner models | Barber et al. 2021 | comparison | **not covered by either paper's theorem**; CV+ alone: >= 0.757 at 90%, 0.557 at 80% |
| CQR score with method B | Romano et al. 2019 | secondary | as B |
| Per-group (Mondrian) calibration: gender (F, M), race (White, Asian) | Romano et al. 2020 | groups | >= target within each group |
| Joint over horizons: alpha / 4 per horizon (Bonferroni); max-normalised score as an extra | Stankeviciute et al. 2021 | horizons | Bonferroni: >= target jointly |

- Scores are whole numbers 0-6, so each set is a set of scores, reported by its coverage and by how
  many of the 7 scores it contains ("values"). No randomised tie-breaking, so coverage can be above
  target. Coverage is averaged over students (each counts once, matching the guarantee); 95% CIs
  from 1,000 student-level bootstrap resamples with the sets held fixed.
- **Pre-registered flags:** *under-coverage* = coverage CI entirely below target; *group gap* = the
  paired CI of female - male or White - Asian excludes zero. **Reliable** = no under-coverage flag
  overall (pooled) and none within female, male, White, Asian (per-group calibration).

```bash
python test_conformal.py
python test_rq3_training.py
python train_forecast.py --run_all --models noode --versions H B Hn --outers 0 1 2 3 4 --roles split --out_root runs/rq3
python train_forecast.py --run_all --models noode_q --versions H --outers 0 1 2 3 4 --roles split --out_root runs/rq3
python train_forecast.py --run_all --models noode --versions H B --outers 0 1 2 3 4 \
    --roles inner0 inner1 inner2 inner3 --out_root runs/rq3
python train_forecast.py --run_all --models ode --versions H --outers 0 1 2 3 4 --roles split --out_root runs/rq3
python baseline_history.py --rq3 --outers 0 1 2 3 4 --out_root runs/rq3
python rq3_conformal.py
```

**Results: method B, pooled calibration** (range over the 2 tasks x 4 horizons; 170 students)

| Model | Coverage at 90% | Values at 90% | Coverage at 80% | Values at 80% | Reliable |
|---|---|---|---|---|---|
| Persistence | 0.931-0.968 | 3.77-3.98 | 0.833-0.954 | 2.47-3.97 | YES |
| Linear (history) | 0.910-0.918 | 2.68-3.50 | 0.808-0.822 | 2.04-2.60 | YES |
| KAN no-ODE H | 0.913-0.923 | 2.90-3.58 | 0.809-0.820 | 2.15-2.68 | YES |
| KAN no-ODE B | 0.908-0.919 | 3.05-3.58 | 0.804-0.816 | 2.21-2.73 | YES |
| KAN no-ODE Hn (no gender / race) | 0.913-0.920 | 2.85-3.58 | 0.807-0.820 | 2.12-2.64 | YES |
| **KAN-ODE H** | 0.914-0.920 | 2.88-3.64 | 0.812-0.823 | 2.13-2.67 | YES |
| Linear quantile (CQR) | 0.936-0.945 | 3.08-3.83 | 0.854-0.888 | 2.50-3.21 | YES |
| Quantile KAN H (CQR) | 0.903-0.920 | 3.02-3.62 | 0.804-0.826 | 2.28-2.83 | YES |

No model had an under-coverage flag (0 of 16 checks overall and 0 of 64 within groups each).
Persistence over-covers (up to 0.97 at 90%) because its errors are whole numbers, so its sets jump
in steps of whole scores.

**Methods compared** (all point models, both tasks, all horizons):

| Method | Under-coverage flags at 90% | at 80% | Mean coverage at 90% | Mean values at 90% |
|---|---|---|---|---|
| **B (main)** | **0 of 64** | **0 of 64** | 0.923 | 3.36 |
| B1 single subsample | 2 of 48 | 0 of 48 | 0.916 | 3.33 |
| A survey-level | **11 of 48** | **20 of 48** | 0.899 | 3.10 |
| Pooled CDF | 0 of 48 | 0 of 48 | 0.911 | 3.22 |
| B + CV+ (empirical only) | 0 of 32 | 0 of 32 | 0.922 | 3.35 |

Treating every survey as an independent point (method A) gives narrower sets that fall short of
the target in 11 checks at 90% and 20 at 80%: students who answer many surveys dominate the
calibration. Calibrating by student (B) fixes this. Pooled CDF reaches the target with slightly
smaller sets but has only an asymptotic guarantee; B + CV+ behaved like B.

**Gender and race** (method B, 90%; full tables in `runs/rq3/eval/rq3_table.md`, not in git):
- *Pooled calibration:* no group falls below 90% (0 flags within female, male, White, Asian). For
  **anxiety**, female students are covered 2-5 points less than male students (e.g. KAN no-ODE H at
  4 weeks: female 0.899, male 0.942, gap -0.044 [-0.075, -0.014]); this gap is flagged at 2-8 weeks
  for most models. Depression and race show no flagged gaps under pooled calibration.
- *Per-group calibration:* both groups reach >= 90% as guaranteed (female about 0.91, male about
  0.95; White about 0.93, Asian about 0.95), but the gaps do **not** close: the smaller groups (male,
  Asian) end up over-covered, because whole-number scores make their thresholds jump. Per-group
  calibration guarantees each group's minimum, not equal coverage.
- The linear quantile model (CQR) shows no flagged gender or race gaps, at the cost of wider sets
  (+0.19 to +0.49 values vs the plain linear model, coverage about 0.94).
- *Sensitivity (Hn, no gender / race inputs):* the same coverage and the same gap pattern as H,
  with equal or marginally smaller sets (0.00 to -0.07 values). The gaps do not come from using gender or
  race as inputs.
- Small groups, pooled calibration (too small to conclude): Black (8 students) 0.88-0.93,
  Hispanic / other (6) 0.97-0.99, multiracial / native (9) 0.90-0.93 for KAN no-ODE H.

**Set sizes (method B, 90%, paired):**
- KAN no-ODE H vs persistence: **0.3-0.9 fewer values** (CIs below zero at every horizon), with
  coverage still above target.
- KAN no-ODE H vs linear: KAN sets are 0.05-0.26 values **wider**; the simple model is at least as
  good here.
- KAN no-ODE B vs H: B wider at 1-2 weeks (+0.05 to +0.15), same later (sensing adds nothing).
- **KAN-ODE H vs KAN no-ODE H: practically identical** (-0.02 to +0.06 values; coverage within 0.004).
- CQR did not give narrower sets than the absolute-residual method for the same model.

**Joint coverage over 1 / 2 / 4 / 8 weeks** (90%): Bonferroni guarantees it but is conservative
(joint coverage 0.96-0.99, about 4.9-5.9 values per horizon). The max-normalised score (an extra,
not from the six papers) gives joint coverage of about 0.91 over the available horizons with about
3.9-4.4 values; on cutoffs that have all 4 horizons it is 0.885-0.906. No joint result has an
under-coverage flag.

**What this means:** conformal prediction can wrap any of these forecasters and give honest 90% or
80% sets for a new student, but only if the calibration is done per student. The sets are wide
(roughly 3 of 7 possible scores at 90%), which reflects how uncertain future PHQ-2 / GAD-2 scores
are; neither the KAN nor the ODE makes them meaningfully narrower than a linear model. Coverage is
not equal across gender for anxiety: pooled calibration covers female students slightly less,
and per-group calibration lifts everyone above the target without equalising coverage.

**The guarantee we claim:** for a new student from the same population and a randomly chosen survey
of that student, the method-B set contains the true score with probability >= 1 - 2 alpha (observed:
about 1 - alpha); with per-group calibration, the same within female, male, White and Asian students.
It is **not** a guarantee for a specific student, a specific survey, the latest survey, or other
universities and years.

**Limitations:** cross-validation on 170 students, not an untouched test set (already used twice);
students assumed to be drawn from one population although cohorts span 2017-2022 including COVID;
surveys of one student are not exchangeable, so the guarantee is for a random survey of a new
student; whole-number scores cause over-coverage; one seed; inputs normalised with the original
149 training students; models use gender and race as inputs (Hn check: no change); small groups
too small to conclude.
