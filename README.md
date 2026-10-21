# kan-digital-phenotyping

Research code for *Longitudinal, Causally-Informed Multitask Digital Phenotyping with
Uncertainty-Aware KAN Models* (B.Tech HCI + AI project). This is a research tool, not a
clinical diagnostic system.

**Current stage:** re-implementation of **Interpretable-MTLNet** (Ali, Ni & Kundur, NeurIPS 2025
workshop) aligned to the College Experience Study (CES) dataset. Detection setting, same as the
paper: predict current depression / anxiety screen status from the 60 days of phone sensing
before a survey. Later stages (forecasting with KAN-ODE, factor analysis, conformal prediction)
will build on this model.

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
