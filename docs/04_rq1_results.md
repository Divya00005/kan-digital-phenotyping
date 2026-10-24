# 04 · RQ1 results: metrics explained, every table, and what it means

> Code: `evaluate_forecast.py` (metrics, bootstrap, decision rule, test lock),
> `evaluate_history_control.py`, `baseline_history.py`.
> Result files: `runs/final/eval_test/rq1_table.md` (variant A, test),
> `runs/variantC/final/eval_test/rq1_table.md` (variant C, test),
> `runs/historyctl/eval/table.md` (history-only control, cross-validation). All git-ignored.

## 1. The metrics, simply

### MAE (mean absolute error) — "how far off is the predicted score?"
Average of |predicted − true| over all forecasts. Model scores are clipped to 0–6 first.
Example: true 2, predicted 1.4 → error 0.6. **Primary metric of RQ1**, averaged over depression
and anxiety ("MAE avg"). Lower is better. **Mistake to avoid:** MAE ignores *who* is at risk; a
model can have bad MAE and still rank people well.

### Persistence baseline — "future score = today's score"
The simplest honest forecaster. Scores change slowly, so it is hard to beat. If a model cannot beat
persistence, it adds nothing for predicting the *number*.

### Mean-score baseline
Always predict the training students' average future score. Any model worse than this has learned
nothing.

### AUROC — "does the model rank the right people higher?"
Pick one student who *will* screen positive and one who will not; AUROC = probability the model
gives the first a higher risk. 0.5 = coin flip, 1.0 = perfect. Computed from the screen-positive
logit (for persistence: from today's score). **Mistake:** AUROC can look good when positives are
easy to spot from today's score alone — that is why we compare with persistence.

### AUC-PR — "when the model raises an alarm, how often is it right?"
Area under the precision–recall curve. Better than AUROC for **rare** events. **No-skill level =
prevalence** (e.g. 14.6% positives → random guessing gives about 0.146).

### Onset (early warning)
Only cutoffs where today's score is **below 3**; positive = score **≥ 3 at the horizon**. Onset is
rare: 5.6–9.4% (test set). Onset AUC-PR tells whether the model can catch *new* cases — persistence
cannot predict onset at all except by ranking today's sub-threshold scores (a 2 is more likely to
become 3 than a 0).

### Student-level bootstrap and 95% CI (`evaluate_forecast.bootstrap`)
Resample the **students** (not the surveys) with replacement 1,000 times, recompute the metric each
time, take the 2.5th and 97.5th percentiles. **Why by student:** one student's surveys are
correlated; resampling surveys would make the CI far too narrow.

### Paired difference and its CI
In each bootstrap sample, compute metric(model 1) − metric(model 2) **on the same students**. The CI
of that difference tells whether one model is *reliably* better. If it contains 0, we cannot say.

### Pre-registered decision rule (fixed in `CLAUDE.md` before results)
"The ODE helps" only if the paired 95% CI of MAE(ODE) − MAE(control) is **entirely below 0 at ≥ 2
of the 4 horizons**. Same style for variant C and (with AUROC) for the history control.

### Seed averaging
Each model trained with seeds 0–4; scores averaged, screen logits turned into probabilities and
averaged.

## 2. Variant A (test set, 43 students, used once on 2026-10-22 09:06)

Source: `runs/final/eval_test/rq1_table.md`. Lock: `runs/TEST_SET_USED.json`.

### Version B (sensing + symptom history) — MAE averaged over depression and anxiety

| Horizon | n (students) | Mean score | Persistence | KAN no-ODE | KAN-ODE | KAN-ODE − no-ODE |
|---|---|---|---|---|---|---|
| 1 week | 3965 (43) | 1.152 | **0.633** [0.535, 0.725] | 0.754 [0.636, 0.906] | 0.739 [0.624, 0.880] | −0.015 [−0.030, −0.005] |
| 2 weeks | 3909 (43) | 1.159 | **0.715** [0.601, 0.820] | 0.788 [0.666, 0.937] | 0.785 [0.667, 0.926] | −0.004 [−0.017, 0.004] |
| 4 weeks | 3799 (43) | 1.174 | **0.805** [0.680, 0.920] | 0.855 [0.719, 1.017] | 0.864 [0.735, 1.017] | +0.009 [−0.005, 0.019] |
| 8 weeks | 3636 (42) | 1.195 | **0.917** [0.754, 1.060] | 0.937 [0.785, 1.105] | 0.946 [0.801, 1.107] | +0.009 [−0.006, 0.023] |

Decision: better only at 1 week → **ODE helps: NO**. Model − persistence: no-ODE +0.121
[0.028, 0.273] at 1 week (worse), +0.074 / +0.050 / +0.019 at 2/4/8 weeks (CIs include 0).

### Version A (sensing only) — MAE averaged

| Horizon | Mean score | Persistence | KAN no-ODE | KAN-ODE | KAN-ODE − no-ODE |
|---|---|---|---|---|---|
| 1 week | 1.152 | **0.633** | 1.140 | 1.138 | −0.002 [−0.014, 0.012] |
| 2 weeks | 1.159 | **0.715** | 1.143 | 1.142 | −0.001 [−0.014, 0.014] |
| 4 weeks | 1.174 | **0.805** | 1.156 | 1.157 | +0.001 [−0.012, 0.017] |
| 8 weeks | 1.195 | **0.917** | 1.181 | 1.178 | −0.002 [−0.016, 0.016] |

**ODE helps: NO.** Both models ≈ the mean-score baseline (differences −0.012 to −0.017, CIs include
0): **phone sensing alone carries almost no forecasting signal**.

### Secondary metrics, version B (persistence / KAN no-ODE / KAN-ODE)

| Horizon | Task (prevalence, onset rate) | AUROC | AUC-PR | Onset AUC-PR |
|---|---|---|---|---|
| 1w | depression (14.6%, 5.6%) | 0.894 / 0.906 / 0.907 | 0.649 / 0.720 / 0.727 | 0.127 / 0.226 / 0.226 |
| 1w | anxiety (13.4%, 6.1%) | 0.864 / 0.882 / 0.885 | 0.572 / 0.657 / 0.659 | 0.117 / 0.245 / 0.252 |
| 2w | depression (14.4%, 6.7%) | 0.865 / 0.889 / 0.891 | 0.571 / 0.665 / 0.669 | 0.149 / 0.273 / 0.270 |
| 2w | anxiety (13.8%, 6.7%) | 0.849 / 0.877 / 0.879 | 0.544 / 0.634 / 0.640 | 0.123 / 0.258 / 0.258 |
| 4w | depression (15.0%, 7.4%) | 0.847 / 0.872 / 0.870 | 0.546 / 0.629 / 0.621 | 0.148 / 0.257 / 0.257 |
| 4w | anxiety (13.6%, 7.3%) | 0.814 / 0.853 / 0.853 | 0.484 / 0.602 / 0.600 | 0.119 / 0.217 / 0.218 |
| 8w | depression (15.7%, 9.2%) | 0.810 / 0.830 / 0.830 | 0.486 / 0.576 / 0.576 | 0.173 / 0.268 / 0.270 |
| 8w | anxiety (14.8%, 9.4%) | 0.742 / 0.792 / 0.794 | 0.411 / 0.526 / 0.532 | 0.121 / 0.204 / 0.207 |

Paired CIs (computed during the RQ1 write-up from `rq1_results.json`): KAN-ODE beats persistence on
AUROC at 6 of 8 horizon–task pairs and on onset AUC-PR at all 8; KAN-ODE vs no-ODE AUROC differences
are all within ±0.003 with CIs including 0.

Version A secondary: AUROC 0.53–0.62 for both KAN models (vs persistence 0.74–0.89); onset AUC-PR
0.06–0.12, **below** persistence.

## 3. Variant C (test set, second use, 2026-10-22 11:25)

Source: `runs/variantC/final/eval_test/rq1_table.md`. Lock: `runs/TEST_SET_USED_variantC.json`.

| Horizon | Persistence | Variant A KAN-ODE (ref) | GRU-Δt | KAN ODE-RNN | ODE-RNN − GRU-Δt |
|---|---|---|---|---|---|
| 1 week | **0.633** | 0.739 | 0.763 [0.649, 0.910] | 0.766 [0.648, 0.922] | +0.003 [−0.007, 0.015] |
| 2 weeks | **0.715** | 0.785 | 0.805 [0.687, 0.947] | 0.805 [0.684, 0.957] | +0.000 [−0.009, 0.012] |
| 4 weeks | **0.805** | 0.864 | 0.880 [0.750, 1.028] | 0.880 [0.745, 1.037] | −0.000 [−0.010, 0.012] |
| 8 weeks | **0.917** | 0.946 | 0.962 [0.816, 1.115] | 0.961 [0.809, 1.122] | −0.001 [−0.011, 0.012] |

**ODE-RNN helps: NO.** Both beat persistence on AUROC at 7/8 pairs and onset AUC-PR at 8/8, at the
same level as variant A. Two secondary differences favoured the control slightly (anxiety onset
AUC-PR at 1 and 4 weeks, −0.010); with 16 comparisons this is consistent with chance.

## 4. History-only control (cross-validation, 170 students, test set not used)

Source: `runs/historyctl/eval/table.md`. Version H = version B model with sensing inputs set to 0.

**Primary: screen-positive AUROC averaged over depression and anxiety**

| Horizon | Persistence | Linear/logistic on history | KAN H (history) | KAN B (sensing + history) | B − H |
|---|---|---|---|---|---|
| 1w | 0.882 | **0.910** | 0.907 | 0.902 | −0.006 [−0.010, −0.002] |
| 2w | 0.852 | **0.886** | 0.885 | 0.881 | −0.004 [−0.008, −0.000] |
| 4w | 0.832 | 0.867 | 0.867 | 0.865 | −0.002 [−0.007, 0.004] |
| 8w | 0.801 | 0.840 | **0.842** | 0.840 | −0.002 [−0.007, 0.004] |

Decision: **sensing adds ranking information: NO** (B is *worse* at 1–2 weeks).

| Horizon | Onset AUC-PR: persistence / linear / H / B | MAE avg: persistence / linear / H / B |
|---|---|---|
| 1w | 0.142 / 0.270 / 0.266 / 0.255 | **0.559** / 0.600 / 0.628 / 0.671 |
| 2w | 0.150 / 0.283 / 0.279 / 0.267 | **0.644** / 0.678 / 0.685 / 0.718 |
| 4w | 0.146 / 0.276 / 0.272 / 0.267 | **0.711** / 0.740 / 0.736 / 0.754 |
| 8w | 0.146 / 0.257 / 0.260 / 0.255 | **0.781** / 0.796 / 0.793 / 0.807 |

KAN H vs linear: AUROC differences ≤ 0.003, CIs include 0. Version B stopped improving earlier
(median best epoch 4 vs 9 for H): sensing mostly added room to overfit.

## 5. Plain answers

| Question | Answer | Evidence |
|---|---|---|
| Does the KAN-ODE help? | **No.** Neither ODE after the encoder (A) nor inside it (C) | decision rule NO in both test tables |
| Does sensing help? | **No measurable help.** Alone ≈ mean baseline; added to history it slightly *hurts* ranking and MAE | variant A version A; history-control B − H |
| Does any model predict the score better than persistence? | **No.** Worse at 1 week, tied at 2–8 weeks | MAE tables |
| Does any model rank risk better than persistence? | **Yes**, models using symptom history: AUROC +0.02 to +0.05 | secondary tables; history control |
| Does early warning (onset) work? | **Partly.** Onset AUC-PR roughly doubles (≈ 0.13 → ≈ 0.25) but stays low in absolute terms: most alarms would be false | onset columns |
| Do KANs beat a simple model? | **No.** Linear/logistic on history matches KAN H | history control |

## 6. What is proven and what is not

- **Shown (on CES, these designs):** the ODE did not improve forecasts; persistence is not beaten on
  MAE; history-based models rank risk better than persistence; sensing adds nothing measurable.
- **Not shown:** that ODEs never help (other designs, larger data, other targets could differ); that
  sensing has *no* information (only "no evidence"); that rankings would hold in another university
  or year; anything causal.
- **Statistical caveats:** 43 test students → wide CIs; the test set was used twice (designs differ,
  both pre-registered); multiple secondary comparisons → a few "significant" ones are expected by
  chance.
