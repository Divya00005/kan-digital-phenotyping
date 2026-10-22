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

## Environment (this laptop)
- Drive C: is almost full: keep everything on D:. Project: `D:\projects\kan-digital-phenotyping`.
- Python 3.11 venv in `.venv` (torch 2.11 + CUDA 12.8, RTX 4050 6 GB). Before pip or long runs set
  TEMP, TMP and PIP_CACHE_DIR to `D:\tmp`; install with `pip install --no-cache-dir`.
- Run `data_prep.py` / `forecast_prep.py` from `data/` (outputs `data/*.npz`). One KAN-ODE epoch
  is ~2-2.5 min on the GPU (batch 512); the no-ODE model ~7 s.

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

## Next step
Variant C, step 2: add the past-survey arrays to `forecast_prep.py` (pre-registration above).
Ask the student before starting. Not chosen yet: a symptom-history-only control; RQ2 / RQ3.
Work in major steps and stop after each one for approval, unless the student approves a block.
