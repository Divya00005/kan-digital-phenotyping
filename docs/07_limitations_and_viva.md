# 07 · Limitations, what we can claim, and viva preparation

## 1. Limitations and threats to validity

| Area | Limitation | Effect / how it was handled |
|---|---|---|
| Sample size | 213 students; 43 test, 22 validation | wide CIs; validation-based tuning was noisy (validation CIs ≈ ±0.15 MAE) |
| Test-set reuse | test set used twice (variant A, then variant C) | both uses pre-registered and locked; variant C was designed knowing variant A's result (reported) |
| Population | one university, 2017–2022 incl. COVID; students assumed drawn from one population | results may not transfer to other campuses / years; conformal guarantees rely on this assumption |
| Labels | PHQ-2 / GAD-2 screening scores, self-reported, whole numbers 0–6 | not diagnoses; ties cause conformal over-coverage |
| Irregular data | surveys ≈ every 5 days; targets matched within ±3 days | some horizons missing (only 12,106 of 22,699 train+val cutoffs have all 4) |
| Sensing quality | missingness 6% → 42% over the years; iOS / Android differences | cross-platform features only, invalid days masked; sensing still added nothing |
| Normalisation | CV experiments use arrays normalised with the original 149 training students | inputs only, no labels; equal for all compared models; reported, not fixed |
| Seeds | RQ1: 5 seeds; RQ3: 1 seed | seed SD ≈ 0.01 MAE; RQ3 widths could shift slightly with other seeds |
| Within-student dependence | surveys of one student are correlated | splits, bootstraps and RQ3 calibration all done by student |
| Multiple comparisons | many secondary metrics | only pre-registered primary rules decide; a few "significant" secondary results expected by chance |
| Model coverage of the question | only two ODE designs; no MLP comparison | "the ODE did not help **in these designs**" |
| Missing file | detection result files (`runs/seed0/`) not on this laptop | detection numbers come from `README.md` only |
| Doc / data mismatch | `CLAUDE.md` demographic counts (F 146, M 69, multiracial 14) are for the full table; forecasting data has F 142, M 68, multiracial 13 | explained in `01_data.md` |
| Not done | RQ1 plan mentioned "MLP-ODE / 2×2 comparison later" | never run; would be part of a model ablation |

## 2. What we can and cannot claim

**Can claim (with evidence):**
1. In CES, a KAN-ODE did not forecast PHQ-2 / GAD-2 better than the same KAN without an ODE — for
   an ODE after the encoder (variant A) and inside it (variant C). Pre-registered rules: NO, NO.
2. No model beat persistence on score error (MAE); history-based models ranked future
   screen-positive and onset risk better than persistence.
3. That ranking gain comes from symptom history; adding phone sensing gave no measurable benefit,
   and a linear / logistic model on history was as good as the KAN.
4. Conformal prediction with student-level calibration (Dunn-style repeated subsampling) reached 90%
   and 80% coverage for all models, overall and within female, male, White and Asian students
   (nested CV); survey-level calibration did not.
5. Under pooled calibration, anxiety coverage was 2–5 points lower for female than male students.

**Cannot claim:**
- That ODEs or KANs never help; that sensing has no information; any causal effect of behaviour;
  clinical usefulness; transfer to other populations; per-student coverage; equal coverage across
  groups; anything about small racial groups; a diagnosis from any output.

## 3. How RQ1 and RQ3 connect

```
RQ1: which model forecasts best?           -> no model beats persistence on MAE; history models rank better
History control: where does signal come from? -> symptom history, not sensing
RQ3: how uncertain are the forecasts?       -> wrap any RQ1 model with conformal sets
     and are those uncertainty ranges fair? -> reliable with student-level calibration; wide; anxiety gender gap
```
RQ3 does not need a "best" model: conformal works around any predictor. It therefore turns RQ1's
modest point forecasts into honest ranges, and shows the ranges are about equally wide for KAN,
KAN-ODE and linear models — consistent with RQ1's finding that the extra machinery adds no signal.

## 4. What remains

- **RQ2** (behavioural factors linked to future worsening) — not done. The supervisor suggested an
  **ablation study** instead. A model-only ablation was planned in conversation (base KAN no-ODE H:
  KAN → MLP, splines off, grid 3 / 10, LayerNorm off, score-only loss, focal → BCE; 5-fold CV, test
  set not used) but **not pre-registered or started**.
- Possible RQ3 follow-ups noted in `CLAUDE.md`: more equal subgroup coverage (CQR showed no gender
  gap), coverage by symptom level.
- Any further test-set use must be agreed and pre-registered.

## 5. 25 likely viva / supervisor questions

1. **What is the project in one sentence?** Forecasting students' future depression / anxiety
   screening scores from phone sensing and past surveys with KAN and KAN-ODE models, and giving
   reliable uncertainty ranges with conformal prediction.

2. **What are PHQ-2 and GAD-2?** Two-item screening scores (0–6) for depression and anxiety; ≥ 3 is
   screen-positive. They are screens, not diagnoses.

3. **Why split by student and not by survey?** Surveys of one student are correlated; a random
   survey split would put the same person in train and test and inflate results.

4. **How did you prevent future-information leakage?** Sensing window ends the day before the survey;
   history uses surveys up to t0 only; targets asserted to be in the future; normalisation from
   training students only; student-disjoint splits asserted in code.

5. **What is a KAN, in simple words?** A network with a learnable curve (a B-spline) on every
   connection, and nodes that just add. MLPs have fixed curves and learnable weights.

6. **Why use B-splines?** Smooth, local bumps whose weights can be learned by gradient descent; inside
   the grid they sum to 1 (tested).

7. **What does the ODE add?** It moves a learned 16-number state forward in time with dz/dt = KAN(z),
   so the model can be asked about any day ahead.

8. **Why RK4 and not Euler?** Much more accurate per step (worked example: error 0.00024 vs 0.107);
   our RK4 matched an adaptive solver to 1.7e-6.

9. **Why a no-ODE control?** To separate the effect of the ODE from the effect of extra KAN layers;
   it has the same encoder and a parameter-matched KAN block (144,624 vs 144,336 parameters).

10. **Did the KAN-ODE help?** No. Variant A: ODE better only at 1 week (−0.015 MAE), rule needs 2
    horizons → NO. Variant C: differences ≤ 0.003 → NO.

11. **Did any model beat persistence?** Not on MAE (persistence 0.633 at 1 week vs 0.739 best KAN-ODE).
    Yes on ranking: e.g. 1-week depression AUROC 0.907 vs 0.894; onset AUC-PR ≈ 0.25 vs ≈ 0.13.

12. **Why is persistence so hard to beat?** Scores change slowly and are mostly 0–2; today's score is
    already the best single guess of the number.

13. **Does phone sensing help?** Not measurably. Sensing only ≈ mean-score baseline; adding it to
    history slightly lowered AUROC at 1–2 weeks (B − H −0.006, −0.004; CIs below 0).

14. **Is that a failure of the project?** No — it is a clear negative result obtained with
    pre-registered rules, and it agrees with the weak detection result (AUROC 0.583).

15. **Why AUROC and AUC-PR, not accuracy?** Positives are ~12–15% (onset 5–9%); accuracy would look
    high by predicting "negative" always. AUC-PR's no-skill level equals the prevalence.

16. **What is the student-level bootstrap?** Resample students 1,000 times to get CIs; paired
    differences use the same resampled students for both models.

17. **What is pre-registration and why did you do it?** Writing the plan and decision rule in
    `CLAUDE.md` and committing it before running, so the rules cannot be changed after seeing results.

18. **What is conformal prediction?** A wrapper that uses held-out calibration errors to build ranges
    that contain the truth with a guaranteed probability, for any model.

19. **Why did survey-level conformal fail?** Surveys of one student are not exchangeable; heavy
    responders dominate the calibration quantile. It under-covered in 11 of 48 checks at 90% and 20
    of 48 at 80%.

20. **What is your exact RQ3 guarantee?** For a new student and a random survey of theirs, P(true
    score in set) ≥ 1 − 2α with method B (observed ≈ 1 − α); with per-group calibration, the same
    within each of female, male, White, Asian.

21. **Why did you not use the test set for RQ3?** It had been used twice and has no Black or
    Hispanic and only 9 Asian students; nested CV on 170 students gives 4× more students per group.

22. **What does per-group (Mondrian) calibration do, and did it make coverage equal?** It calibrates
    within each group, guaranteeing each group's minimum. All groups reached ≥ 90%, but coverage was
    not equal (male / Asian ≈ 0.95 vs female ≈ 0.91) because small groups' thresholds jump in steps.

23. **Is the anxiety gender gap caused by using gender as an input?** No evidence of that: the model
    without gender / race inputs (Hn) showed the same pattern.

24. **How wide are the intervals, and is that useful?** About 3 of the 7 possible scores at 90%
    (≈ 2.0–2.8 at 80%). Honest but wide; useful as "low / moderate / higher range", not as a precise
    number. Joint 4-horizon sets with Bonferroni are even wider (≈ 5–6 values).

25. **What would you do next?** A model ablation (KAN vs MLP, splines, grid size, loss choices) as the
    supervisor suggested; possibly CQR or other methods for more equal subgroup coverage; and
    validation on another cohort before any practical use.
