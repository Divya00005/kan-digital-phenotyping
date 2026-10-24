# 01 · The data: CES, the audit, and how samples are built

> Code: `data_prep.py` (detection dataset), `forecast_prep.py` (forecasting dataset).
> The raw data are **not in git** (dataset licence). Shapes and counts below come from running
> read-only commands on `data/ces_kan_dataset.npz` and `data/ces_forecast.npz`, and from the
> script outputs recorded during the project.

## 1. The College Experience Study (CES)

- **What:** a 5-year study (2017-09 to 2022-07) of Dartmouth students (Nepal et al. 2024, IMWUT).
  Students installed a phone app that passively logged behaviour and asked short surveys.
- **Three files we use** (`data/`):
  - `EMA/general_ema.csv` — survey answers (EMA = ecological momentary assessment)
  - `Sensing/sensing.csv` — one row per student per day of phone-derived features (~520 MB)
  - `Demographics/demographics.csv` — gender and race only
- `CLAUDE.md` records 218 students with PHQ-4 answers. After the processing rules below, the
  detection dataset keeps **214 students / 27,925 samples** (from 35,348 surveys), and the
  forecasting dataset **213 students / 27,432 cutoffs** (script outputs; README).

## 2. PHQ-4, PHQ-2, GAD-2 and "screen-positive"

The PHQ-4 is four questions, each answered 0 ("not at all") to 3 ("nearly every day"):

| Item in CES | Asks about | Belongs to |
|---|---|---|
| `phq4-1` | feeling nervous, anxious, on edge | **GAD-2** (anxiety) |
| `phq4-2` | not being able to stop worrying | **GAD-2** |
| `phq4-3` | little interest or pleasure | **PHQ-2** (depression) |
| `phq4-4` | feeling down, depressed, hopeless | **PHQ-2** |

`data_prep.py → load_labels()`:

```python
e["anx_score"] = e["phq4-1"] + e["phq4-2"]          # GAD-2, 0..6
e["dep_score"] = e["phq4-3"] + e["phq4-4"]          # PHQ-2, 0..6
e["y_dep"] = (e["dep_score"] >= CUTOFF)              # CUTOFF = 3 -> screen-positive
```

- **Why recompute?** The audit found the provided `phq4_score` disagrees with the items in 182 rows,
  so the subscores are rebuilt from the items.
- **Screen-positive (≥ 3)** is the standard cut-off for PHQ-2 and GAD-2. About **12.3% (depression)
  and 12.5% (anxiety)** of cutoffs are screen-positive today (forecasting data, read-only check).
- **Common mistake:** calling a positive screen "depression". It only means "worth a closer look".

Distribution of future scores in train + validation (all horizons and tasks; checked during RQ3
planning): 0 → 73,191; 1 → 25,521; 2 → 33,985; 3 → 6,573; 4 → 6,417; 5 → 1,214; 6 → 3,243.
**Most scores are 0–2.** This matters later: whole-number, mostly-low scores make some metrics and
the conformal sets behave in "steps".

## 3. Sensing features and the audit

### 3.1 Which daily features (`data_prep.py`, list `DAILY`)
21 cross-platform features in five groups:

| Group | Features |
|---|---|
| activity | `act_still_ep_0`, `act_in_vehicle_ep_0`, `act_on_bike_ep_0` |
| mobility | `loc_dist_ep_0`, `loc_visit_num_ep_0`, `loc_max_dis_from_campus_ep_0` |
| places (time at) | `loc_home_dur`, `loc_self_dorm_dur`, `loc_study_dur`, `loc_social_dur`, `loc_food_dur`, `loc_workout_dur`, `loc_other_dorm_dur`, `loc_leisure_dur` |
| phone use | `unlock_num_ep_0`, `unlock_duration_ep_0`, `unlock_num_ep_1`, `unlock_duration_ep_1` (ep_1 = night) |
| sleep | `sleep_duration`, `sleep_start`, `sleep_end` |

### 3.2 The platform problem
About 88% of days come from iPhones (`CLAUDE.md`). Some features exist only on Android or only on
iOS; on the other platform they are zero-filled. A model could then learn "zero = Android", and
because Android users skew male / Asian, that would be a hidden demographic shortcut. **Fix:** only
features that exist on both platforms are used, and the share of iPhone days (`ios_share`) is kept
as an explicit static input.

### 3.3 Valid days and missingness (`load_sensing()`)
```python
s["valid"] = (quality_activity >= 12 h) & (quality_loc >= 12 h)
s.loc[s.valid == 0, DAILY] = np.nan          # an invalid day's zeros mean "not measured"
s.loc[(sleep_duration < 1) | (sleep_duration > 16), "sleep_duration"] = np.nan
```
- **Why:** a day with only 3 hours of phone data would show "0 km travelled", which is not true.
- Missingness rises from 6% (2017) to 42% (2022) (`CLAUDE.md`, audit).
- Heavy-tailed counts / durations are log-transformed: `np.log1p` for all non-sleep features (`LOG_FEATS`).

### 3.4 Demographics (`static_table()`)
- Gender: F, M, "both", unknown → 3 indicator columns (`gender_female`, `gender_male`,
  `gender_other_or_unknown`).
- Race → 6 indicator columns (white, asian, black, hispanic_other, multiracial_native, unknown).
- Plus `ios_share` → **S has 10 columns**.
- **Counts differ by file — not an error, a different population:** `CLAUDE.md` lists the full
  demographics table (F 146, M 69, multiracial/native 14). In the **forecasting data (213
  students)** the counts are F 142, M 68, "both" 1, unknown 2; White 133, Asian 51, Black 8,
  Hispanic/other 6, multiracial/native 13, unknown 2 (read-only check during RQ3 planning).

## 4. Detection dataset: `data_prep.py → main()`

One sample per (student, survey day t). Steps:

1. **Window:** days t-60 … t-1 — the survey day itself is **excluded** (`sl = slice(end - WINDOW, end)`).
2. **Keep rule:** at least 7 valid days among the last 14 (`MIN_VALID_LAST14 = 7`).
3. **Engineered features `E` (34)** from `engineered()`: for 7 key features — 14-day mean and
   coefficient of variation over 3 / 7 / 14 days (7 × 4 = 28); weekend-minus-weekday for 4 features
   (4); share of valid days in the last 14 and 60 days (2). 28 + 4 + 2 = 34.
4. **Student-level split 70 / 10 / 20** with `default_rng(42)`: **149 / 22 / 43 students**
   (19,839 / 3,260 / 4,826 samples; read-only check).
5. **Normalisation with training students only:** z-score with training mean / SD, missing values
   → training median (in z units), clip to [-5, 5]; then append the valid-day flag as a 22nd channel.

Result (`ces_kan_dataset.npz`): `X (27925, 60, 22)`, `E (27925, 34)`, `S (27925, 10)`, `Y (27925, 2)`.

## 5. Forecasting dataset: `forecast_prep.py → main()`

### 5.1 Cutoffs
A **cutoff t0 = a survey day**. "Today the student filled in a survey; forecast their scores 1, 2, 4
and 8 weeks later." Same 60-day window and keep rule as detection. A cutoff is dropped if it has no
future survey at any horizon.

### 5.2 Targets: `targets()`
```python
goal = t0 + 7*h days                                   # h = 1, 2, 4, 8 weeks
ok = (|survey_date - goal| <= 3 days) & (survey_date > t0)
take the survey closest to goal; store dep, anx and the exact offset in days
```
- **Why ±3 days:** surveys are irregular (about every 5 days), so "exactly 7 days later" rarely exists.
- **Why store the exact offset:** the KAN-ODE integrates to the real day (e.g. 15, not 14).
- `mask[i, h] = False` if no survey fell in the window. Targets available: 23,286 / 23,107 /
  22,469 / 21,519 cutoffs at 1 / 2 / 4 / 8 weeks; only 12,106 of the 22,699 train + val cutoffs
  have all four.

### 5.3 Symptom history features `H` (6): `history_features()`
Known on day t0 (the survey of day t0 itself is included, nothing after):

| Feature | Meaning |
|---|---|
| `dep_today`, `anx_today` | today's PHQ-2, GAD-2 |
| `dep_mean_28d`, `anx_mean_28d` | mean of surveys in the last 28 days (incl. today) |
| `days_since_prev_survey` | gap to the previous survey (capped at 60) |
| `n_surveys_28d` | number of surveys in the last 28 days |

`current` = today's two raw scores, used by persistence and by the "onset" metric.

### 5.4 Past-survey sequence (variant C only): `survey_sequence()`
`surv_scores (N, 61, 2)` = PHQ-2 / GAD-2 ÷ 6 on each survey day in t0-60 … t0, zero elsewhere;
`surv_mask (N, 61)` marks survey days. Added later (commit e537253); all 18 older arrays were checked
**bit-for-bit unchanged** and the new arrays reproduce `H` exactly (error ~1e-7).

### 5.5 Split and normalisation
The forecasting dataset **reuses the detection split** (`--split_from ces_kan_dataset.npz`), so the
same 149 / 22 / 43 students. Normalisation of `X`, `E`, `H` again uses training students only.
Final shapes (read-only check of `data/ces_forecast.npz`):

| Array | Shape | Meaning |
|---|---|---|
| `X` | (27432, 60, 22) | 60 days × 21 features + valid flag, normalised |
| `E` | (27432, 34) | engineered summaries |
| `S` | (27432, 10) | gender, race, iPhone share |
| `H` | (27432, 6) | symptom history (normalised) |
| `score` | (27432, 4, 2) | future dep / anx at 4 horizons (0 where missing) |
| `mask` | (27432, 4) | which horizons have a target |
| `offset_days` | (27432, 4) | exact days ahead |
| `current` | (27432, 2) | today's raw scores |
| `surv_scores`, `surv_mask` | (27432, 61, 2), (27432, 61) | past surveys (variant C) |
| `uid`, `date`, `split`, `gender`, `race` | (27432,) | bookkeeping |

## 6. Input versions

| Version | Inputs given to the model | Where it is set |
|---|---|---|
| **A** | X, E, S (sensing + static) | `train_forecast.py → inputs()` |
| **B** | X, E, S, **H** | same |
| **H** | H, S; X and E replaced by **zeros** | same (history-only control) |
| **Hn** | H; X, E zeros; the 9 gender/race columns of S zeroed (`ios_share` kept) | same (RQ3 sensitivity) |

"Zeros" rather than deleting the inputs keeps the architecture and parameter count identical, so a
difference between versions can only come from the information, not from model size.

## 7. How future-information leakage is prevented

| Risk | Protection | Where |
|---|---|---|
| Sensing from the survey day or later | window is t0-60 … t0-1 | `slice(end - WINDOW, end)` |
| Future surveys in history | only `lab_dates <= t0` | `history_features()`, `survey_sequence()` |
| Target not in the future | `assert np.all(offset_days[mask] > 0)` | `forecast_prep.py` |
| Same student in train and test | split by student; `assert` | `forecast_prep.py`, `train_forecast.load_data()` |
| Test statistics in normalisation | mean / SD / median from training students only | both prep scripts |
| Test set in RQ3 / history control | code asserts no split-2 row is used | `train_forecast.py`, `rq3_conformal.py` |

**Known, reported caveat:** in the cross-validation experiments (history control, RQ3) the arrays
were normalised with the original 149 training students, some of whom are later held out. This
uses inputs only (no labels) and affects all compared models equally.

## 8. One sample, from raw data to model input

Read-only inspection of training row 859 of `ces_forecast.npz` (student ID omitted):

```
cutoff date t0          : 2020-06-26  (a survey day)
today (dep, anx)        : (2, 2)  -> below the cut-off 3, so this row counts for "onset"
sensing window          : 2020-04-27 .. 2020-06-25 (60 days); valid days: 11 of 60
                          (kept because >= 7 of the last 14 days are valid)
targets found           : offsets 7, 15, 26, 55 days  (closest surveys to 7, 14, 28, 56 within +/-3)
future (dep, anx)       : 1w (1, 4)  2w (1, 1)  4w (2, 1)  8w (0, 0)
mask                    : [True, True, True, True]
surveys in 61-day window: 4 (day t0 included)
S                       : [1,0,0, 1,0,0,0,0,0, 0.0] -> female, White, iPhone share 0 (Android)
```

Shapes as the model sees them for one sample (`train_forecast.inputs()`, version B):

```
x  (60, 22)  ->  KANConv1d k=3,5,7  ->  average over days  ->  64
e  (34,)     ------------------------------------------------>  34   ] concat (104) -> LayerNorm
h  (6,)      ------------------------------------------------>   6   ]  -> KANLinear -> 64
s  (10,)     ->  KANLinear -> 8 ----------------------------------------> concat 72 -> z(t0) (16)
offset_days (4,) = [7, 15, 26, 55]  -> used by the ODE / no-ODE head (file 03)
output (4, 4)  = per horizon [dep score, anx score, dep logit, anx logit]
```

Note the anxiety target at 1 week is **4 (screen-positive)** while today is 2: an "onset" event at
1 week that returns below 3 by week 2 — a reminder of how noisy week-to-week scores are.
