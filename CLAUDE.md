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

## RQ1 plan (agreed)
- Models: persistence, KAN no-ODE (control), **KAN-ODE variant A** (encoder -> z(t0) ->
  dz/dt = KAN(z) -> heads). Variant C (KAN ODE-RNN encoder, Rubanova et al. 2019) only if A
  shows no ODE benefit. MLP-ODE / 2x2 comparison later.
- Input versions: A = sensing only; B = sensing + symptom history known at t0.
- Targets: future scores (MAE) + screen-positive (AUROC, AUC-PR) + onset among currently < 3.
- Evaluation per horizon with student-level bootstrap CIs and paired CIs for differences.
- **Decision rule (fixed in advance):** "the ODE helps" only if the paired CI of
  KAN-ODE vs no-ODE excludes zero at >= 2 horizons.

## Next step
Write `train_forecast.py` (GPU if available; `--model {ode,noode} --version {A,B} --seed N`;
`--run_all --seeds ...`) and an evaluation script for the RQ1 table. Then 4 development runs,
then 5 seeds. Ask the student before starting.
