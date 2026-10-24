# 05 · Conformal prediction from the basics

> Code that implements these ideas: `conformal.py` (`split_threshold`, `p_values`, `k_upper`,
> `abs_candidates`, `sets_from_threshold`, `covered`). Tests on synthetic data: `test_conformal.py`.

## 1. Why uncertainty?

A forecast "your depression score in 2 weeks will be 1.4" sounds precise, but RQ1 showed typical
errors of 0.6–0.9 points. A counsellor needs to know **how sure** the forecast is.

**Everyday example:** a weather app saying "70% chance of rain" or "18–23 °C" is more useful than
"20 °C", because you can plan for the range.

## 2. Prediction intervals and prediction sets

- **Prediction interval:** a range [low, high] that should contain the true future value.
- Our scores are **whole numbers 0–6**, so we report a **prediction set**: the list of scores the
  range contains, e.g. [0.3, 2.5] → {1, 2}.
- **Coverage** = how often the true value lands in the set. A **90% set** should cover ≥ 90% of cases.
- **Width / number of values** = how informative it is. A set {0,…,6} always covers but says nothing.
  Good uncertainty = **reach the coverage target with the smallest sets**.

## 3. The conformal idea in four steps

**Why conformal:** it works around **any** trained model (KAN, linear, persistence), without
assuming errors are Gaussian, and gives a **guarantee** that holds in finite samples.

1. **Train** a model on one group of data.
2. On a **separate calibration group** (not used for training), compute how wrong the model was:
   a **nonconformity score** for each case, e.g. s = |y − ŷ| (absolute residual).
3. Take a high **quantile** of those scores, about the 90th percentile, with a small correction:
   the **k-th smallest score, k = ⌈(n + 1)(1 − α)⌉**. Call it t.
4. For a new case: **set = all y with |y − ŷ| ≤ t**, i.e. [ŷ − t, ŷ + t].

**Why (n + 1)?** The new case is "one more" point. If all n + 1 scores are exchangeable, the new
score is equally likely to land in any of the n + 1 rank positions, so it is below the k-th smallest
calibration score with probability at least k / (n + 1) ≥ 1 − α.

## 4. Exchangeability (the one assumption)

**Exchangeable** = the order of the data points does not matter statistically; e.g. independent
draws from the same population. Conformal's guarantee needs the calibration cases and the new case
to be exchangeable.

**Where it breaks in our data:** one student's surveys are correlated (same person, overlapping
targets). A heavy responder with 300 surveys is not 300 independent people. If we calibrate on all
surveys, heavy responders dominate the quantile, and the guarantee for a **new student** is lost.
This is the core reason for the student-level methods in file 06 (Dunn et al.). `test_conformal.py →
test_student_level_vs_survey_level` shows it on synthetic data: survey-level calibration covered a
new student only 0.54 of the time, student-level subsampling 0.90.

## 5. Worked example by hand (split conformal)

Calibration residuals |y − ŷ| for **n = 9** calibration cases:
`0.2, 0.5, 0.4, 1.1, 0.3, 0.9, 1.6, 0.7, 0.6`

Target **80%** (α = 0.2):

1. Sort: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, **1.1**, 1.6
2. k = ⌈(9 + 1) × 0.8⌉ = ⌈8⌉ = 8 → the 8th smallest = **t = 1.1**
3. New case, model predicts ŷ = 1.4 → interval [1.4 − 1.1, 1.4 + 1.1] = **[0.3, 2.5]**
4. Possible scores inside: **{1, 2}** → the set has **2 values**.
5. If the true score turns out to be 2 → covered; if 3 → missed.

At **90%** (α = 0.1): k = ⌈10 × 0.9⌉ = 9 → t = 1.6 (the largest) → [−0.2, 3.0] → clipped to [0, 3]
→ {0, 1, 2, 3}. With fewer than 9 calibration points, k would exceed n and the set would be all of
0–6: **small calibration sets give useless intervals**.

**A floating-point trap (handled in `conformal.k_upper`):** in Python, 10 × 0.9 = 9.000000000000002,
so a naive ceil gives 10. The code subtracts a tiny epsilon; `test_quantile_index` checks n = 9 → 9th.

The same rule as a **p-value**: p(y) = (1 + #{calibration scores ≥ s(y)}) / (n + 1); y is in the
set if p(y) > α. `test_pvalue_set_equals_quantile_set` checks the two rules give identical sets.

## 6. What the guarantee says — and does not say

- **Says (marginal coverage):** averaged over random calibration sets and random new cases,
  P(true value in set) ≥ 1 − α.
- **Does not say:** that each individual person, each group, or each type of case is covered 90%.
  That is **conditional coverage**, which needs extra methods (per-group calibration, file 06).
- **Upper bound:** with continuous scores, coverage ≤ 1 − α + 1/(n + 1) (Lei et al.). With
  **whole-number** scores, many ties occur, and coverage can be noticeably **above** the target
  (`test_integer_outcomes_over_cover`: 0.95 at a 90% target). The project reports this instead of
  using randomised tie-breaking.

## 7. Other conformity scores

- **Absolute residual** |y − ŷ| → symmetric interval around the prediction (the default here).
- **CQR score** max(q_lo − y, y − q_hi) for a model that predicts a low and a high quantile → the
  interval [q_lo − t, q_hi + t] can be wider for uncertain cases and narrower for easy ones (file 06).

## 8. Common mistakes

1. Calibrating on the training data → residuals too small → under-coverage.
2. Calibrating per survey when the unit is a person → no guarantee for a new person.
3. Reading "90% coverage" as "90% for every subgroup".
4. Judging only coverage: a set {0..6} covers 100% and is useless — always report width.
5. Tuning anything on the cases used to *evaluate* coverage.
