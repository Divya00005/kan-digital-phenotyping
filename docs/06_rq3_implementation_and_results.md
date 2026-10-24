# 06 · RQ3: how it was implemented, and every result

> Code: `conformal.py` (methods), `quantile_model.py` (quantile KAN, pinball loss),
> `train_forecast.py` (`rq3_students`, `rq3_rows`, versions H / Hn, model `noode_q`),
> `baseline_history.py --rq3` (linear and quantile-linear), `rq3_conformal.py` (evaluation).
> Tests: `test_conformal.py` (15, synthetic data), `test_rq3_training.py` (7).
> Results: `runs/rq3/eval/rq3_table.md`, `runs/rq3/eval/summary.json`, `runs/rq3/eval/all_results.csv`
> (git-ignored). Pre-registration: `CLAUDE.md` (commit d5faf4e), amendment 251e369.

## 1. The question

Can conformal prediction give **reliable** prediction sets for future PHQ-2 / GAD-2 scores at 1, 2,
4 and 8 weeks, **overall and within gender and race groups**? "Reliable" was defined before any
result (section 6).

## 2. Design: nested student-level cross-validation (test set not used)

**Why not the test set:** it had already been used twice (RQ1 variants A and C), and it has no Black
or Hispanic students and only 9 Asian students, too few for subgroup analysis.

**Why nested:** each evaluation student must be unseen by the model **and** by the calibration data.

```
170 train+val students  --cv_folds (default_rng(0))-->  5 outer folds x 34 students
For outer fold k (34 EVALUATION students), use only the other 136 students:
  split role : shuffle (default_rng(3000+k)) -> 68 model students (61 train + 7 early-stop)
                                              + 68 CALIBRATION students        -> main method
  inner roles: shuffle (default_rng(1000+k)) -> 4 inner folds x 34; model j trains on 3 folds
               (92 train + 10 early-stop), predicts inner fold j and the evaluation students -> CV+
Every student is evaluated exactly once (pooled over the 5 outer folds = 170 students, 22,699 cutoffs).
```

Implemented in `train_forecast.rq3_students` / `rq3_rows`; `test_rq3_training.test_nested_folds`
checks sizes, disjointness, no test students and reproducibility.

## 3. Models wrapped (point and quantile predictors)

| Name in tables | What | Runs |
|---|---|---|
| Persistence | future = today's score | no training |
| Linear (history) | `LinearRegression` on [H, S] | `baseline_history.py --rq3` |
| KAN no-ODE H / B | `NoODEForecaster`, versions H and B | split + 4 inner per fold |
| KAN no-ODE Hn | H without the 9 gender / race inputs (sensitivity) | split only |
| **KAN-ODE H** | `KANODEForecaster`, version H (added by amendment) | split only |
| Quantile KAN H (CQR) | `QuantileNoODEForecaster`: 5/10/90/95% outputs, pinball loss | split only |
| Linear quantile (CQR) | `QuantileRegressor` (alpha 0, highs) on [H, S] | split only |

65 KAN / KAN-ODE runs (2.0 h) + 30 linear fits, seed 0, all finished without NaN (`CLAUDE.md`
process notes; `runs/rq3/train.log`).

**Pinball loss (`quantile_model.pinball_loss`):** for quantile level q and error u = y − prediction,
loss = max(q·u, (q − 1)·u). For q = 0.9, under-predicting costs 9× more than over-predicting, so the
best constant is the 90th percentile. `test_pinball_loss` checks the minimiser is the empirical
quantile. Quantiles are sorted after prediction so they never cross.

## 4. The methods and where they come from

The prompt's labels A–F map to the code as follows (the code's own "A" and "B" names are kept in the
tables):

| Prompt | Code name | What | Paper | Guarantee |
|---|---|---|---|---|
| A | **A** | survey-level split conformal: k-th smallest of **all calibration surveys'** residuals | Lei et al. 2018, Alg. 2 | none for a new student (surveys not exchangeable) |
| B | **B (MAIN)** | one random survey per calibration student, repeated 200×; average the p-values; keep y if mean p > α | Dunn, Wasserman & Ramdas 2023, repeated subsampling | **≥ 1 − 2α** for a random survey of a new student; in practice ≈ 1 − α |
| B | B1 | the same with a single subsample | Dunn et al. | ≥ 1 − α (exact) |
| B | pCDF | each student's residual distribution weighted equally | Dunn et al., Method 1 | asymptotic only |
| C | **BCV** | B combined with CV+ over the 4 inner models | Barber et al. 2021 + Dunn | **not covered by either theorem** (empirical only) |
| D | CQR | method B with score max(q_lo − y, y − q_hi) | Romano, Patterson & Candès 2019 | as B |
| E | pooled vs per-group | per-group = calibrate within female / male, White / Asian separately | Romano, Barber, Sabatti & Candès 2020 | ≥ target **within each group** |
| F | joint | Bonferroni: α/4 per horizon | Stankeviciute, Alaa & van der Schaar 2021 | ≥ target jointly |
| extra | maxnorm | max over horizons of \|y − ŷ\| / s_h | not from the six papers | labelled extra |

### 4.1 Method B in detail (`conformal.subsample_threshold`)
```
for b in 1..200:
    pick 1 random survey per calibration student (one_per_student)  -> m scores
    p_b(y) = (1 + #{scores >= s(y)}) / (m + 1)
set = { y in 0..6 : mean_b p_b(y) > alpha }
```
**Why one per student:** then the m calibration scores are like m independent people, exchangeable
with "a random survey of a new student". **Why repeat:** one draw wastes most surveys and is noisy;
averaging 200 draws is stable but the theory only guarantees 1 − 2α (averaging p-values costs a
factor 2). The code computes the threshold t* once and the set is {y : s(y) ≤ t*};
`test_subsample_threshold_matches_definition` checks this shortcut against the definition.

### 4.2 CV+ (`conformal.cvplus_bounds`, `cvplus_subsample_sets`)
For a new point, each calibration point i contributes μ₋fold(i)(x) ± Rᵢ, using the model that did not
see i. Lower bound = ⌊α(n+1)⌋-th smallest of μ − R, upper = ⌈(1−α)(n+1)⌉-th smallest of μ + R
(Barber eq. 11; `test_cvplus_brute_force`). Guarantee for one subsample: 1 − 2α − min{2(1 − 1/K) /
(n/K + 1), (1 − K/n)/(K + 1)}; with n = 136, K = 4 this is 1 − 2α − 0.043 → **≥ 0.757 at 90%,
≥ 0.557 at 80%**. Combined with 200 subsamples and a majority vote, no theorem applies.

### 4.3 Per-group (Mondrian) calibration (`conformal.group_thresholds`)
Compute the threshold using only calibration students of the same group; apply it to evaluation
students of that group. Two separate partitions: gender (F, M) and race (White, Asian). Smaller
groups (Black 8, Hispanic/other 6, multiracial/native 9, race unknown 1, gender "both" 1, gender
unknown 1) are reported under pooled calibration only, as "too small to conclude".

### 4.4 Joint coverage (`conformal.joint_covered`, `max_norm_score`)
A cutoff is jointly covered if **every** available horizon is covered. Bonferroni: run method B at
α/4 per horizon (union bound). Reported over available horizons and on the 12,106 cutoffs with all 4.

### 4.5 Evaluation (`rq3_conformal.py`)
- Coverage per cutoff → averaged **per student first, then over students** (`stats`): each student
  counts once, matching the guarantee. Cutoff-weighted coverage also stored.
- "values" = number of the 7 scores in the set; width = clipped interval width.
- 1,000 student bootstrap resamples (seed 0), sets held fixed; gaps and model differences paired.
- A fresh `default_rng(4000 + k)` for every combination (amendment 3), so order does not matter.
- Bookkeeping cross-checked by recomputing three coverages by hand (identical; `CLAUDE.md`).

## 5. The exact guarantee we claim — and do not claim

**Claim:** for a **new student** from the same population and a **randomly chosen survey** of that
student, the method-B set contains the true PHQ-2 / GAD-2 score with probability **≥ 1 − 2α**
(observed ≈ 1 − α); with per-group calibration, the same holds **within** female, male, White and
Asian students separately.

**Do not claim:** coverage for a specific student, for each individual survey, for a student's
latest survey, for other universities / years, or equal coverage across groups. Pooled calibration
never guarantees coverage inside a group.

## 6. Pre-registered flags and verdict

- **Under-coverage flag:** the 95% CI of coverage lies entirely below the target.
- **Group-gap flag:** the paired CI of female − male (or White − Asian) excludes 0.
- **Reliable (per model):** method B has no under-coverage flag overall (pooled) **and** none within
  F, M, White, Asian under per-group calibration — both targets, both tasks, all horizons.

## 7. Results

### 7.1 Main table: method B, pooled calibration, 90% target
Coverage · number of values (source: `runs/rq3/eval/rq3_table.md`; 170 students):

| Model | Task | 1 wk | 2 wk | 4 wk | 8 wk |
|---|---|---|---|---|---|
| Persistence | dep | 0.968 · 3.77 | 0.957 · 3.77 | 0.945 · 3.77 | 0.931 · 3.77 |
| Persistence | anx | 0.968 · 3.98 | 0.954 · 3.97 | 0.945 · 3.97 | 0.931 · 3.97 |
| Linear (history) | dep | 0.914 · 2.68 | 0.913 · 2.89 | 0.917 · 3.18 | 0.918 · 3.37 |
| Linear (history) | anx | 0.910 · 2.93 | 0.913 · 3.17 | 0.914 · 3.30 | 0.915 · 3.50 |
| KAN no-ODE H | dep | 0.921 · 2.90 | 0.922 · 3.16 | 0.923 · 3.33 | 0.922 · 3.46 |
| KAN no-ODE H | anx | 0.915 · 3.09 | 0.914 · 3.28 | 0.913 · 3.35 | 0.915 · 3.58 |
| KAN no-ODE B | dep | 0.919 · 3.05 | 0.916 · 3.21 | 0.919 · 3.34 | 0.916 · 3.48 |
| KAN no-ODE B | anx | 0.908 · 3.21 | 0.911 · 3.33 | 0.911 · 3.35 | 0.912 · 3.58 |
| KAN no-ODE Hn | dep | 0.916 · 2.85 | 0.917 · 3.12 | 0.917 · 3.25 | 0.920 · 3.42 |
| KAN no-ODE Hn | anx | 0.913 · 3.07 | 0.914 · 3.23 | 0.914 · 3.34 | 0.917 · 3.58 |
| **KAN-ODE H** | dep | 0.918 · 2.88 | 0.920 · 3.16 | 0.919 · 3.33 | 0.919 · 3.50 |
| **KAN-ODE H** | anx | 0.918 · 3.08 | 0.915 · 3.26 | 0.914 · 3.35 | 0.919 · 3.64 |
| Linear quantile (CQR) | dep | 0.939 · 3.08 | 0.940 · 3.24 | 0.940 · 3.37 | 0.936 · 3.56 |
| Linear quantile (CQR) | anx | 0.945 · 3.42 | 0.945 · 3.58 | 0.942 · 3.66 | 0.941 · 3.83 |
| Quantile KAN H (CQR) | dep | 0.919 · 3.02 | 0.920 · 3.14 | 0.916 · 3.25 | 0.920 · 3.39 |
| Quantile KAN H (CQR) | anx | 0.914 · 3.23 | 0.911 · 3.33 | 0.903 · 3.40 | 0.913 · 3.62 |

At 80%: KAN and linear models 0.804–0.826 with 2.0–2.8 values; persistence jumps between 0.83 and
0.95 because its residuals are whole numbers (thresholds move in steps of 1); linear CQR 0.85–0.89.

**Meaning:** every model reaches the target; the sets contain about **3 of 7 scores** at 90%.
(Illustration only, not a specific student from the data: a 90% set of 3 values could look like
{0, 1, 2} for a prediction near 1, and a 4-value set like {1, 2, 3, 4} for a prediction near 2.5.)

### 7.2 Methods compared (all point models, both tasks, all horizons; `summary.json` + CSV)

| Method | Under-coverage flags at 90% | at 80% | Mean coverage 90% | Mean values 90% |
|---|---|---|---|---|
| **B (main)** | **0 of 64** | **0 of 64** | 0.923 | 3.36 |
| B1 single subsample | 2 of 48 | 0 of 48 | 0.916 | 3.33 |
| A survey-level | **11 of 48** | **20 of 48** | 0.899 | 3.10 |
| pooled CDF | 0 of 48 | 0 of 48 | 0.911 | 3.22 |
| B + CV+ (empirical) | 0 of 32 | 0 of 32 | 0.922 | 3.35 |

**Meaning:** ignoring that surveys belong to students (A) gives narrower sets that **fall short** —
heavy responders dominate. Student-level B never falls short. For persistence, A, pooled CDF, B and
B + CV+ give identical rows (its whole-number residuals lead to the same threshold); only the single
subsample B1 differs slightly (e.g. depression 2 weeks 0.933 instead of 0.957).

### 7.3 Gender and race (method B, 90%, KAN no-ODE H shown; all models in `rq3_table.md`)

Coverage female / male (gap, 95% CI); ⚑ = group-gap flag:

| Calibration | Task | 1 wk | 2 wk | 4 wk | 8 wk |
|---|---|---|---|---|---|
| pooled | dep | 0.917 / 0.931 (−0.014) | 0.919 / 0.931 (−0.011) | 0.922 / 0.926 (−0.004) | 0.920 / 0.926 (−0.006) |
| pooled | anx | 0.909 / 0.927 (−0.018) | 0.903 / 0.940 (−0.037) ⚑ | 0.899 / 0.942 (−0.044 [−0.075, −0.014]) ⚑ | 0.904 / 0.940 (−0.036) ⚑ |
| per-group | dep | 0.913 / 0.964 ⚑ | 0.912 / 0.963 ⚑ | 0.913 / 0.966 ⚑ | 0.911 / 0.961 ⚑ |
| per-group | anx | 0.917 / 0.936 | 0.915 / 0.947 ⚑ | 0.911 / 0.945 ⚑ | 0.915 / 0.949 ⚑ |

Race, KAN no-ODE H, White / Asian: pooled — no flagged gap (e.g. 1 wk dep 0.919 / 0.922);
per-group — Asian over-covered (≈ 0.95) vs White ≈ 0.92–0.94, one flag (anxiety 1 wk).

**Meaning:**
- Under pooled calibration **no group falls below 90%** (0 flags within F, M, White, Asian), but for
  **anxiety, female students are covered 2–5 points less** than male students, flagged at 2–8 weeks
  for most models.
- Per-group calibration lifts **both** groups above 90% (as guaranteed) but **does not equalise**:
  the smaller groups (male: 51 students, Asian: 42) end up over-covered (~0.95), because with
  whole-number scores their thresholds jump in steps. Per-group calibration guarantees a minimum,
  not equality.
- The **linear CQR** model showed no flagged gender or race gap, at the cost of wider sets.
- **Hn** (no gender / race inputs) shows the same pattern → the gaps do not come from using these
  inputs.
- Small groups (pooled, KAN H): Black (8 students) 0.88–0.93, Hispanic/other (6) 0.97–0.99,
  multiracial/native (9) 0.90–0.93 — too small to conclude.

### 7.4 Model comparisons (method B, pooled, 90%; paired)

| Comparison | Δ values (range over horizons, both tasks) | Meaning |
|---|---|---|
| KAN H − persistence | −0.87 to −0.31 (all CIs < 0) | KAN sets clearly narrower |
| KAN H − linear | +0.05 to +0.26 (CIs > 0) | linear sets slightly narrower |
| KAN B − KAN H | +0.15 / +0.12 at 1 wk, ≈ 0 later | sensing makes sets wider, not narrower |
| KAN Hn − KAN H | −0.07 to +0.00 | removing demographics changes almost nothing |
| **KAN-ODE H − KAN H** | −0.02 to +0.06; coverage within 0.004 | **the ODE changes nothing** |
| Quantile KAN − KAN H | +0.14 at 1 wk, CIs include 0 later | CQR not narrower here |
| Linear CQR − linear | +0.19 to +0.49 | CQR wider (but no gender gap) |

### 7.5 Joint coverage over 1/2/4/8 weeks (90%; `summary.json`)

| Method | Joint coverage | Values per horizon | Under-coverage flags |
|---|---|---|---|
| Bonferroni (α/4 each), available horizons | 0.967–0.986 | 4.88–5.92 | 0 |
| Bonferroni, cutoffs with all 4 horizons | 0.957–0.985 | 4.87–5.91 | 0 |
| max-normalised (extra), available horizons | 0.911–0.920 | 3.88–4.42 | 0 |
| max-normalised (extra), all 4 horizons | 0.885–0.906 | 3.87–4.43 | 0 |

**Meaning:** Bonferroni is safe but wasteful (≈ 5–6 of 7 scores). The extra max-normalised score
reaches ≈ 90% with smaller sets; it is reported as an extra because it is not from the six papers.

### 7.6 Verdict (`summary.json → verdict`)

| Model | Reliable | Under-coverage overall | Within groups | Mean coverage − 90% |
|---|---|---|---|---|
| Persistence | **YES** | 0 of 16 | 0 of 64 | +0.050 |
| Linear (history) | **YES** | 0 of 16 | 0 of 64 | +0.014 |
| KAN no-ODE H | **YES** | 0 of 16 | 0 of 64 | +0.018 |
| KAN no-ODE B | **YES** | 0 of 16 | 0 of 64 | +0.014 |
| KAN no-ODE Hn | **YES** | 0 of 16 | 0 of 64 | +0.016 |
| KAN-ODE H | **YES** | 0 of 16 | 0 of 64 | +0.018 |
| Linear quantile (CQR) | **YES** | 0 of 16 | 0 of 64 | +0.041 |
| Quantile KAN H (CQR) | **YES** | 0 of 16 | 0 of 64 | +0.015 |

No group set was trivial (all of 0–6) in any fold (column "Trivial group sets" = 0).

## 8. Plain answer to RQ3

**Yes — conformal prediction gives reliable 90% and 80% sets for a new student, overall and within
female, male, White and Asian students, but only when calibration treats the student as the unit.**
The sets are wide (≈ 3 of 7 scores), neither the KAN nor the ODE makes them meaningfully narrower
than a linear model, and coverage is not equal across gender for anxiety.

## 9. What is not proven

- Coverage on a fresh, untouched test set (this is nested CV on 170 students).
- Coverage for other cohorts / years (students assumed from one population; 2017–2022 includes COVID).
- Per-student or per-survey coverage (the guarantee is marginal over random surveys of new students).
- Anything about small racial groups.
- Why the anxiety gender gap exists (the data cannot tell; it is not explained by using gender as input).
