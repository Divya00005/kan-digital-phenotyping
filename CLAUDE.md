# Project context for Claude

B.Tech HCI + AI research project: *Longitudinal, Causally-Informed Multitask Digital Phenotyping
with Uncertainty-Aware KAN Models*. Research tool, not a clinical system.

## Working rules (from the student)
- **Ask before implementing each step.** Explain the plan, wait for "yes", then build.
- Explain in simple English; the student is learning KAN, ODEs, causal inference, conformal prediction.
- Never invent variables or labels; never use future information as input; never call screening
  scores a diagnosis; never claim KAN/KAN-ODE beats a baseline without evidence; report negative
  results; split train/val/test by student; tune only on validation; test set used once.
- Never commit data (`*.csv`, `*.npz`, `*.zip` are git-ignored). End commits with the
  Co-Authored-By line used in the existing history.

## Research questions
- **RQ1**: can a KAN-ODE predict future depression/anxiety symptoms (1, 2, 4, 8 weeks ahead)
  better than persistence and than the same KAN without an ODE?  <- current focus
- **RQ2**: which behavioural factors are associated with future worsening; causal claims only
  under explicit assumptions (CES has no physiology, few confounders -> mostly associational).
- **RQ3**: conformal prediction intervals; coverage overall and by gender / race.

## Dataset: College Experience Study (CES), Kaggle
Not in git. Expected local layout: `data/EMA/general_ema.csv`, `data/Sensing/sensing.csv`,
`data/Demographics/demographics.csv`. 218 students with PHQ-4, 2017-09 to 2022-07.

Audit rules (already implemented in `data_prep.py`):
- PHQ-4 items: `phq4-1`, `phq4-2` = anxiety (GAD-2); `phq4-3`, `phq4-4` = depression (PHQ-2).
  Recompute subscores from items (provided `phq4_score` disagrees in 182 rows).
  Screen-positive = subscore >= 3.
- Surveys roughly every 5 days (irregular), not weekly.
- Use cross-platform sensing features only: Android-only / iOS-only features are zero-filled on
  the other platform (88% of days are iOS; Android users skew male/Asian).
- Valid day = quality_activity >= 12 h and quality_loc >= 12 h; invalid days -> missing + mask.
  Missingness rises from 6% (2017) to 42% (2022).
- Demographics: gender (F 146, M 69, 1 "both") and race only. RQ3: female vs male and
  White vs Asian testable; Black (8), Hispanic/other (6), multiracial/native (14) descriptive only.

## Done
1. **KAN-MTLNet reproduction** (Ali, Ni & Kundur 2025) in detection setting: `data_prep.py`,
   `kan.py`, `model.py`, `train.py`, `test_kan.py`. Seed-0 test macro AUROC 0.583
   [0.533, 0.636] (paper 0.731 on All of Us; differences explained in README). Overfits after ~6 epochs.
2. **RQ1 steps 1-2**: `forecast_prep.py` (27,432 cutoffs, 213 students; cutoff = survey day,
   60 days of sensing before it, targets at +1/2/4/8 weeks within +/-3 days, exact day offsets),
   `ode.py` (RK4, 1-day steps), `forecast_model.py` (`KANODEForecaster`, parameter-matched
   `NoODEForecaster`), `test_ode.py` (all pass). Persistence baseline in `baselines_test.json`.

3. **RQ1 variant A, trained and tested** (2026-10-22): `train_forecast.py`, `evaluate_forecast.py`.
   Settings chosen on validation (A: lr 3e-4; B: lr 1e-3, no residual), 5 seeds, predictions averaged.
   **Test result: ODE helps = NO for both versions.** Version B: ODE better than no-ODE only at
   1 week (MAE -0.015 [-0.030, -0.005]); no KAN model beats persistence on MAE (worse at 1 week,
   tied at 2-8 weeks); both KAN models beat persistence on screen-positive AUROC (6/8) and onset
   AUC-PR (8/8), identically, so the gain is not from the ODE. Version A (sensing only) = no better
   than the training mean. Full numbers in README. **The test set has been used once**
   (`runs/TEST_SET_USED.json`); evaluating any new model on it is a second use and must be agreed
   with the student first.

4. **RQ1 variant C, trained and tested** (2026-10-22; pre-registration below): `odernn_model.py`,
   `test_odernn.py` (all pass), past-survey arrays in `forecast_prep.py` (old arrays unchanged),
   variant C options in `train_forecast.py` / `evaluate_forecast.py`. lr 1e-3 chosen on validation.
   **Test result: ODE-RNN helps = NO.** ODE-RNN vs GRU-Δt differs by at most 0.003 MAE at every
   horizon; neither beats persistence (worse at 1 week, tied 2-8 weeks); ranking gains over
   persistence (AUROC 7/8, onset AUC-PR 8/8) are the same as variant A. **RQ1 conclusion so far:
   in CES an ODE did not help, after the encoder (A) or inside it (C).** The test set has now been
   used **twice** (`runs/TEST_SET_USED.json`, `runs/TEST_SET_USED_variantC.json`); any further
   test use must be agreed with the student and reported as such.

5. **History-only control** (2026-10-22; pre-registration below): `--version H` and CV folds in
   `train_forecast.py`, `baseline_history.py`, `evaluate_history_control.py`,
   `test_history_control.py` (all pass). 5-fold CV on the 170 train + val students, test not used.
   **Result: sensing adds ranking information = NO.** AUROC(B) - AUROC(H) is -0.006 / -0.004 /
   -0.002 / -0.002 (B worse at 1-2 weeks, CI below zero); a linear / logistic regression on history
   matches the KAN models; all history models beat persistence on AUROC (+0.02 to +0.04) and onset
   AUC-PR (about x2); no model beats persistence on MAE, and B has the highest MAE of the models.
   **Takeaway: the ranking gain comes from symptom history, not phone sensing.**

## Environment (this laptop)
- Drive C: is almost full: keep everything on D:. Project: `D:\projects\kan-digital-phenotyping`.
- Python 3.11 venv in `.venv` (torch 2.11 + CUDA 12.8, RTX 4050 6 GB). Before pip or long runs set
  TEMP, TMP and PIP_CACHE_DIR to `D:\tmp`; install with `pip install --no-cache-dir`.
- Run `data_prep.py` / `forecast_prep.py` from `data/` (outputs `data/*.npz`). One KAN-ODE epoch
  is ~2-2.5 min on the GPU (batch 512); the no-ODE model ~7 s; ODE-RNN ~45 s; GRU-Δt ~4 s.
- Long runs: launch in the background and keep the session open; `--run_all` skips finished runs,
  so an interrupted job can simply be started again (same seed = same result).

## RQ1 plan (agreed)
- Models: persistence, KAN no-ODE (control), **KAN-ODE variant A** (encoder -> z(t0) ->
  dz/dt = KAN(z) -> heads). Variant C (KAN ODE-RNN encoder, Rubanova et al. 2019) only if A
  shows no ODE benefit. MLP-ODE / 2x2 comparison later.
- Input versions: A = sensing only; B = sensing + symptom history known at t0.
- Targets: future scores (MAE) + screen-positive (AUROC, AUC-PR) + onset among currently < 3.
- Evaluation per horizon with student-level bootstrap CIs and paired CIs for differences.
- **Decision rule (fixed in advance):** "the ODE helps" only if the paired CI of
  KAN-ODE vs no-ODE excludes zero at >= 2 horizons.

## RQ1 variant C: pre-registration (agreed with the student 2026-10-22, before any variant C code)
Question: does an ODE **inside the encoder** help? KAN ODE-RNN (Rubanova et al. 2019) vs a GRU-Δt control.

**Inputs (version B only).** Daily sensing for days t0-60 .. t0-1 (the 22 channels of `X`, incl.
valid mask), plus a new array of past surveys: PHQ-2 / GAD-2 (divided by 6) on each survey day in
t0-60 .. t0, day t0 included (known at t0), nothing after t0. Plus `E`, `H`, `S` as in version B.
`forecast_prep.py` gets the new arrays only; existing arrays must stay identical (27,432 cutoffs,
same baselines).

**Models** (identical except for the ODE; parameter counts matched within 1%):
- *KAN ODE-RNN:* hidden state h (32). Walk day 0 .. 60 (t0). Between days, h evolves by
  dh/dt = KAN(h) (`KANDynamics`, width 32), one RK4 step of 1 day. On a valid sensing day,
  h <- GRU_sensing(h, x_day). On a survey day, h <- GRU_survey(h, scores). Invalid / no-survey
  days: no update.
- *GRU-Δt (control):* same two GRU cells and same update days, but h does not change between
  updates; the days since the last update are an extra GRU input. The same-sized KAN block is
  applied once to the final h (as in the variant A no-ODE control).
- Both: GRU cells are standard `nn.GRUCell`. Final h is combined with `E`, `H`, `S` exactly as in
  the current `Encoder` (no conv branch) -> z(t0) (16) -> the **no-ODE head** of `NoODEForecaster`
  (horizon as input). Loss = `forecast_loss`.

**Training:** AdamW (weight decay 1e-4), batch 512, gradient clipping 1.0, max 100 epochs,
early stopping on validation loss (patience 10). Same for both models.

**Tuning (validation only, seed 0):** lr in {1e-3, 3e-4}, both models. Keep the lr with the lowest
validation MAE (mean over 4 horizons and both tasks, averaged over the two models, so both share
it); ties within 0.005 -> 1e-3. Only allowed extra fix, if a model gives NaN / blows up: lr 1e-4
for both. Anything else -> stop and ask the student.

**Final:** seeds 0-4 per model; predictions averaged over seeds.

**Evaluation:** `evaluate_forecast.py` metrics, 1,000-resample student-level paired bootstrap.
Primary = MAE averaged over depression and anxiety, per horizon. Secondary = per-task MAE, AUROC,
AUC-PR, onset AUROC / AUC-PR. Reference rows (not in the rule): mean score, persistence, variant A
version B no-ODE and KAN-ODE (their existing test predictions).

**Decision rule:** "the ODE-RNN helps" only if the paired 95% CI of
MAE(ODE-RNN) - MAE(GRU-Δt) is entirely below zero at >= 2 of the 4 horizons.

**Test set policy:** this is the **second use** of the test set (first: variant A, 2026-10-22).
Evaluated once, with `--final`, under its own lock `runs/TEST_SET_USED_variantC.json`. The README
must say it is the second use and that the design was informed by variant A's validation and test
results. This section is committed to git **before** any variant C code or training. Changes after
that are allowed only before the test evaluation, only on validation evidence, and only as a dated
amendment below with the reason.

**Major steps (stop after each for approval):** 1 this pre-registration; 2 data arrays;
3 models + `test_odernn.py` + speed check; 4 seed-0 runs and lr choice; 5 seeds 0-4;
6 test evaluation once + README / CLAUDE.md update.

**Amendments:** none.
**Process notes:** the NaN fix was not needed. The step-5 training job was cut off when a session
ended (ODE-RNN seed 2 at epoch 10, nothing saved); it was restarted and only unfinished runs were
trained. Test evaluated once on 2026-10-22 11:25.

## History-only control: pre-registration (agreed with the student 2026-10-22, before any code)
Question: version B models rank screen-positive / onset risk better than persistence. Does that
gain come from **phone sensing**, or only from the **symptom-history** features? Predictive
(associational) question only; no causal claim.

**Students:** the 170 train + validation students of `data/ces_forecast.npz` (22,699 cutoffs).
The test set (43 students) is **not used**: no model predicts split 2 in this experiment.

**Folds:** 5-fold cross-validation by student. Folds = `numpy.random.default_rng(0).permutation`
of the sorted 170 student IDs, cut into 5 near-equal parts (34 each). In each fold, the other
136 students train; 10% of them (14, chosen with `default_rng(100 + fold)`) are held out for
early stopping only. Every student is predicted once, by models that never saw them.

**Models** (variant A `NoODEForecaster`, identical architecture and parameter count):
- *Version B:* sensing `X`, `E` + history `H` + static `S`.
- *Version H:* the same, but `X` and `E` replaced by zeros (all channels, incl. the valid-day
  mask) in training and prediction. `H` and `S` unchanged (`S` = gender, race group and
  `ios_share`, i.e. device platform, not behaviour).
- *Simple baseline (reference):* per horizon and task, on [`H`, `S`] of the fold's training
  students: `LinearRegression` for the score (clipped 0-6) and `LogisticRegression` (C = 1.0,
  max_iter 1000) for screen-positive (>= 3). Rows with a target only.

**Training (no tuning):** variant A final version-B settings: AdamW lr 1e-3, weight decay 1e-4,
batch 512, gradient clipping 1.0, max 100 epochs, early stopping on the early-stopping students'
loss (patience 10), `forecast_loss`. Seeds 0-4 for every fold and version (50 runs). If any run
gives NaN: stop and ask the student.

**Predictions:** per student, average over the 5 seeds; pool the out-of-fold predictions of all
5 folds (170 students). Persistence and mean-score rows as before (mean = training-fold mean).

**Primary metric:** screen-positive **AUROC averaged over depression and anxiety**, per horizon.
**Decision rule:** "sensing adds ranking information" only if the 95% paired student-bootstrap CI
(1,000 resamples, seed 0) of AUROC(B) - AUROC(H) lies entirely **above zero** at **>= 2 of the 4
horizons**. Otherwise the result is "no evidence that sensing adds ranking information" (not
"sensing has no information").
**Secondary (reported, not decisive):** onset AUC-PR, AUC-PR, MAE (all per task and averaged);
B, H and the simple baseline vs persistence; B and H vs the simple baseline; spread over folds and seeds.

**Known caveat (reported, not fixed):** `ces_forecast.npz` was normalised with the original 149
training students, so some held-out students' inputs were part of those means / SDs (inputs only,
no labels; same for B and H).

**Major steps (stop after each for approval):** 1 this pre-registration; 2 code + tests
(`--version H`, fold split, simple baseline; tests: version H output unchanged when `X` / `E`
change, folds student-disjoint and complete, no split-2 rows); 3 runs (50 KAN runs + baseline);
4 evaluation table, decision rule, README / CLAUDE.md update.

**Amendments:** none.
**Process notes:** 50 runs finished in 86 min without errors. To compute the primary metric,
`evaluate_forecast.compute` gained task-averaged keys (`*_auroc_avg`, `*_auprc_avg`,
`*_onset_auprc_avg`); all existing RQ1 numbers were checked unchanged (2,567 of 2,567).

## RQ3: pre-registration (agreed with the student 2026-10-22, before any RQ3 code)
**Question:** can conformal prediction give reliable prediction intervals for future PHQ-2
(depression) and GAD-2 (anxiety) scores at 1, 2, 4 and 8 weeks, overall and within gender and
race groups? Papers (local `papers/`, not in git): Lei et al. 2018 (split conformal), Dunn,
Wasserman & Ramdas 2023 (two-layer hierarchical), Barber et al. 2021 (jackknife+ / CV+), Romano,
Patterson & Candès 2019 (CQR), Romano, Barber, Sabatti & Candès 2020 (equalized coverage),
Stankeviciute, Alaa & van der Schaar 2021 (conformal time-series forecasting).

**Data and test set.** The 170 train + validation students of `data/ces_forecast.npz` only.
**The test set is not used** (already used twice); no model predicts split 2; code asserts this.
Inputs keep the existing normalisation (caveat as in the history control).

**Nested student-level cross-validation.**
- *Outer folds:* the history-control folds (`cv_folds`, 5 x 34 students). Outer fold k = the
  evaluation students; everything else for that round uses only the other 136 students.
- *Split for the main method:* the 136 students, shuffled with `default_rng(3000 + k)`, are cut
  into 68 model students and 68 calibration students (Lei et al.: equal split). The split model
  trains on the 68 model students (7 of them, `default_rng(3100 + k)`, held out for early stopping).
- *Inner folds for CV+:* the same 136 students, shuffled with `default_rng(1000 + k)`, cut into
  4 inner folds of 34. Inner model j trains on the other 3 inner folds (102 students; 10 held out
  for early stopping, `default_rng(2000 + 10k + j)`) and predicts inner fold j and outer fold k.
- Evaluation students are never in any training, early-stopping or calibration set of their round.

**Models (point predictors; one seed, seed 0).**
- *Persistence:* future score = today's score (no training).
- *Linear history model:* `LinearRegression` on [H, S] per horizon and task (as `baseline_history.py`).
- *KAN no-ODE version H* (history; sensing inputs zero) and *KAN no-ODE version B* (sensing +
  history): `NoODEForecaster`, the history-control training recipe (AdamW lr 1e-3, weight decay
  1e-4, batch 512, clipping 1.0, early stopping on loss, patience 10, max 100 epochs).
- *Sensitivity model: version H without demographics ("Hn"):* as H, but the 9 gender / race
  columns of S set to zero (`ios_share` kept).
- *Quantile models for CQR:* (i) a KAN no-ODE version H with quantile outputs at 5 / 95% and
  10 / 90% per horizon and task, trained with the pinball loss only, same recipe, early stopping
  on pinball loss; (ii) linear quantile regression (`QuantileRegressor`, alpha = 0, solver highs)
  on [H, S], same quantiles.
- **KAN-ODE is not included now.** The code reads point predictions per model by name, so
  KAN-ODE can be added later by training its nested runs, without changing the methods.

**Methods** (alpha = 0.10 main, 0.20 also; score values are integers 0-6, so every set is a set
of integers, reported as [lowest, highest] and as the number of score values it contains;
intervals clipped to [0, 6]; **no randomised tie-breaking**: over-coverage is reported).
- **B (MAIN): student-level repeated subsampling** (Dunn et al., repeated subsampling, split
  version). Per horizon and task, B = 200 subsamples (`default_rng(4000 + k)`); in each, one
  cutoff drawn uniformly per calibration student among their cutoffs with a target. Score
  s = |y - y_hat| (CQR: s = max(q_lo - y, y - q_hi)). p_b(y) = (1 + #{c : s_c >= s(y)}) / (m + 1)
  over the m calibration students; set = {y in 0..6 : mean_b p_b(y) > alpha}.
  **Formal guarantee:** for a new student from the same population and a cutoff chosen uniformly
  at random from that student's cutoffs, P(true score in set) >= 1 - 2 alpha (in practice expected
  near 1 - alpha). Also reported for reference: a single subsample (exact >= 1 - alpha).
- **A (reference): survey-level split conformal** (Lei et al. Algorithm 2) on all calibration
  cutoffs, k-th smallest residual with k = ceil((n + 1)(1 - alpha)). No valid new-student guarantee
  (cutoffs of one student are not exchangeable); shown to see what ignoring students does.
- **Pooled-CDF comparison** (Dunn et al. Method 1): each calibration student's residual CDF
  weighted equally; asymptotic only (assumes continuous outcomes, which ours are not).
- **B + CV+ (empirical comparison only):** CV+ intervals (Barber et al. eq. 11) from the 4 inner
  models, one cutoff per calibration student per subsample, final set = score values inside the
  CV+ interval for at least half of the 200 subsamples. **This combination is not covered by
  either paper's theorem.** For one subsample, CV+'s own bound with n = 136 students and K = 4 is
  1 - 2 alpha - min{2(1 - 1/K)/(n/K + 1), (1 - K/n)/(K + 1)} = 1 - 2 alpha - 0.043,
  i.e. >= 0.757 at alpha = 0.10 and >= 0.557 at alpha = 0.20.
- **CQR:** method B with the CQR score on the two quantile models.
- **Pooled vs per-group (Mondrian) calibration** (Romano et al. 2020): per-group = calibration
  scores of the same group only, two separate partitions: gender (female, male) and race (White,
  Asian). Pooled calibration never guarantees coverage within a group. Black (8 students),
  Hispanic / other (6), multiracial / native (9 in these 170), race unknown (1), gender "both" (1)
  and gender unknown (1): pooled analysis only, coverage shown as "too small to conclude".
  If a group has too few calibration students for a non-trivial set, the set is all of 0..6;
  the number of such cases is reported.
- **Joint coverage over horizons** (Stankeviciute et al.): Bonferroni, each horizon at alpha / 4;
  a cutoff is jointly covered if every horizon with a target is covered. Also reported on the
  12,106 cutoffs that have all 4 horizons. **Extra, clearly labelled, not from these papers:**
  max-normalised joint score max_h |y_h - y_hat_h| / s_h, with s_h = the split model's MAE at
  horizon h on its own model students.

**Evaluation** (per method x model x task x horizon, and joint; 170 evaluation students pooled
over the 5 outer folds): coverage averaged over students (primary; matches the guarantee) and
over cutoffs; coverage minus target; mean interval width; mean number of score values;
per group: coverage, width, number of values, students and cutoffs; gaps female - male and
White - Asian. 95% CIs from 1,000 student-level bootstrap resamples (seed 0); gaps and model
differences paired within each resample. Model comparisons: width of H vs persistence, B vs H,
H vs linear, Hn vs H, at the same method and target.
**Pre-registered flags:** *under-coverage* = the CI of coverage lies entirely below the target;
*group gap* = the paired CI of a gap excludes zero. "Reliable" (method B) = no under-coverage flag
overall (pooled calibration) and none within female, male, White, Asian under per-group
calibration. Results are reported whatever they show.

**Files (new unless noted):** `conformal.py` (methods, numpy only), `test_conformal.py`,
`quantile_model.py`, `train_forecast.py` (nested-fold options, version Hn, quantile model),
`rq3_conformal.py` (intervals, evaluation, tables). Outputs in `runs/rq3/` (not in git).

**Major steps (stop after each for approval):** 1 this pre-registration + `papers/` in
`.gitignore`; 2 `conformal.py` + `test_conformal.py` on synthetic data only; 3 nested training
code + tests (folds disjoint and complete, no test rows, Hn ignores demographics, pinball loss);
4 training runs (4 KAN models x 5 outer folds x (1 split + 4 inner) = 100 runs, about 2-3 h, plus
linear models); 5 evaluation and tables; 6 README / CLAUDE.md update.

**Amendments:** none.

## Next step
RQ3, step 2: `conformal.py` and `test_conformal.py` on synthetic data (pre-registration above).
Ask the student before starting. Later decision: whether to add KAN-ODE to RQ3 after the KAN
results. RQ2 is still open.
Work in major steps and stop after each one for approval, unless the student approves a block.
