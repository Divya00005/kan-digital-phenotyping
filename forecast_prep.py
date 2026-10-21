"""
RQ1 step 1: forecasting dataset for KAN-ODE.

Cutoff t0 = a PHQ-4 survey day ("today the student filled in a survey").
  inputs : 60 days of phone sensing BEFORE t0 (days t0-60 .. t0-1), same features as data_prep.py
           + engineered and static features
           + history-score features known at t0 (used only in input version B)
  targets: PHQ-2 (depression) and GAD-2 (anxiety) at t0 + 1, 2, 4, 8 weeks,
           taken from the survey closest to t0 + 7h days within +/- 3 days.
           The exact day offset is stored, so the ODE integrates to the real target time.

Also computes the persistence baseline ("future score = today's score") and the
mean-score baseline on the test students.

Usage:
    python forecast_prep.py --ema general_ema.csv --sensing sensing.csv \
        --demo demographics.csv --split_from ces_kan_dataset.npz --out ces_forecast.npz
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from data_prep import (load_labels, load_sensing, static_table, engineered, engineered_names,
                       DAILY, LOG_FEATS, WINDOW, MIN_VALID_LAST14, CUTOFF)

HORIZONS_WEEKS = [1, 2, 4, 8]
TOLERANCE_DAYS = 3
HIST_NAMES = ["dep_today", "anx_today", "dep_mean_28d", "anx_mean_28d",
              "days_since_prev_survey", "n_surveys_28d"]


def history_features(lab_dates, lab_dep, lab_anx, i):
    """Symptom information available at cutoff i (survey i itself included, nothing after)."""
    t0 = lab_dates[i]
    past = (lab_dates <= t0) & (lab_dates >= t0 - np.timedelta64(28, "D"))
    prev = lab_dates[:i]
    gap = (t0 - prev[-1]).astype(int) if len(prev) else 60
    return [lab_dep[i], lab_anx[i], lab_dep[past].mean(), lab_anx[past].mean(),
            min(gap, 60), past.sum()]


def targets(lab_dates, lab_dep, lab_anx, i):
    """For each horizon: (dep, anx, offset_days) of the survey closest to t0 + 7h, or NaN."""
    t0 = lab_dates[i]
    out = np.full((len(HORIZONS_WEEKS), 3), np.nan, np.float32)
    for k, h in enumerate(HORIZONS_WEEKS):
        goal = t0 + np.timedelta64(7 * h, "D")
        diff = np.abs((lab_dates - goal).astype(int))
        ok = np.where((diff <= TOLERANCE_DAYS) & (lab_dates > t0))[0]
        if len(ok):
            j = ok[np.argmin(diff[ok])]
            out[k] = [lab_dep[j], lab_anx[j], (lab_dates[j] - t0).astype(int)]
    return out


def baseline_metrics(cur, tgt, mask, mean_pred):
    """Three-level metrics for persistence (cur) and mean prediction, per horizon and task."""
    res = {}
    for k, h in enumerate(HORIZONS_WEEKS):
        for t, name in enumerate(["depression", "anxiety"]):
            m = mask[:, k]
            y, c = tgt[m, k, t], cur[m, t]
            pos = (y >= CUTOFF).astype(int)
            neg_now = c < CUTOFF
            onset = pos[neg_now]
            res[f"{h}w_{name}"] = {
                "n": int(m.sum()),
                "mae_persistence": float(np.abs(y - c).mean()),
                "mae_mean": float(np.abs(y - mean_pred[t]).mean()),
                "auroc_persistence": float(roc_auc_score(pos, c)),
                "auprc_persistence": float(average_precision_score(pos, c)),
                "prevalence": float(pos.mean()),
                "onset_n": int(neg_now.sum()), "onset_rate": float(onset.mean()),
                "onset_auroc_persistence": float(roc_auc_score(onset, c[neg_now])),
                "onset_auprc_persistence": float(average_precision_score(onset, c[neg_now])),
            }
    return res


def main(a):
    labels = load_labels(a.ema)
    sens = load_sensing(a.sensing)
    sens_raw = sens.copy()
    for f in LOG_FEATS:
        sens[f] = np.log1p(sens[f].clip(lower=0))

    split_src = np.load(a.split_from)
    split_of = dict(zip(split_src["uid"], split_src["split"]))

    by_uid = {u: g.set_index("date").sort_index() for u, g in sens.groupby("uid")}
    by_uid_raw = {u: g.set_index("date").sort_index() for u, g in sens_raw.groupby("uid")}
    X, M, E, PL, HIST, TGT, UID, DATE = [], [], [], [], [], [], [], []

    for u, lab in labels.sort_values("date").groupby("uid"):
        if u not in by_uid or u not in split_of:
            continue
        lab = lab.drop_duplicates("date")
        ld = lab["date"].values.astype("datetime64[D]")
        ldep, lanx = lab["dep_score"].to_numpy(np.float32), lab["anx_score"].to_numpy(np.float32)
        g = by_uid[u]
        idx = pd.date_range(g.index.min() - pd.Timedelta(days=WINDOW), g.index.max())
        g, graw = g.reindex(idx), by_uid_raw[u].reindex(idx)
        feat, featraw = g[DAILY].to_numpy(np.float32), graw[DAILY].to_numpy(np.float32)
        val = g["valid"].fillna(0).to_numpy(np.float32)
        ios = g["is_ios"].to_numpy(np.float32)
        pos = {d: i for i, d in enumerate(idx.values.astype("datetime64[D]"))}
        for i in range(len(ld)):
            end = pos.get(ld[i])
            if end is None or end - WINDOW < 0:
                continue
            sl = slice(end - WINDOW, end)                  # t0-60 .. t0-1
            if val[sl][-14:].sum() < MIN_VALID_LAST14:
                continue
            tg = targets(ld, ldep, lanx, i)
            if np.all(np.isnan(tg[:, 0])):
                continue                                   # no future survey at any horizon
            X.append(feat[sl]); M.append(val[sl])
            wd = np.array([pd.Timestamp(d).weekday() for d in idx[sl]])
            E.append(engineered(featraw[sl], val[sl], wd))
            PL.append(np.nanmean(ios[sl]) if np.any(~np.isnan(ios[sl])) else np.nan)
            HIST.append(history_features(ld, ldep, lanx, i))
            TGT.append(tg); UID.append(u); DATE.append(ld[i])

    X, M, E = np.stack(X), np.stack(M), np.stack(E)
    HIST, TGT = np.array(HIST, np.float32), np.stack(TGT)
    UID, DATE = np.array(UID), np.array(DATE)
    split = np.array([split_of[u] for u in UID], np.int8)
    tr = split == 0

    # ---- normalisation fitted on training students only (same recipe as data_prep.py) ----
    xtr = X[tr].reshape(-1, X.shape[2])
    mu, sd = np.nanmean(xtr, 0), np.nanstd(xtr, 0) + 1e-6
    med = np.nanmedian((xtr - mu) / sd, 0)
    Xz = np.clip(np.where(np.isnan((X - mu) / sd), med, (X - mu) / sd), -5, 5)
    Xz = np.concatenate([Xz, M[..., None]], axis=2).astype(np.float32)
    emed = np.nanmedian(E[tr], 0)
    E = np.where(np.isnan(E), emed, E)
    Ez = np.clip((E - E[tr].mean(0)) / (E[tr].std(0) + 1e-6), -5, 5).astype(np.float32)
    Hz = ((HIST - HIST[tr].mean(0)) / (HIST[tr].std(0) + 1e-6)).astype(np.float32)
    stat, demo = static_table(a.demo, UID)
    PL = np.array(PL, np.float32)
    PL = np.where(np.isnan(PL), np.nanmedian(PL[tr]), PL)
    S = np.concatenate([stat.to_numpy(np.float32), PL[:, None]], 1).astype(np.float32)

    score = TGT[:, :, :2]                                  # (N, 4, 2) future dep/anx scores
    offset_days = TGT[:, :, 2]                             # (N, 4) exact days ahead
    mask = ~np.isnan(score[:, :, 0])
    cur = HIST[:, :2]                                      # today's dep/anx score

    # ---- leakage checks ----
    assert np.all(offset_days[mask] > 0), "target not in the future"
    assert not (set(UID[split == 0]) & set(UID[split == 2])), "student in train and test"

    print(f"cutoffs: {len(X)} from {len(set(UID))} students")
    for k, h in enumerate(HORIZONS_WEEKS):
        print(f"  {h}w target available: " + "  ".join(
            f"{n} {mask[split == s, k].sum():6d}" for n, s in [("train", 0), ("val", 1), ("test", 2)]))

    te = split == 2
    mean_pred = [np.nanmean(score[tr, :, t]) for t in range(2)]
    res = baseline_metrics(cur[te], score[te], mask[te], mean_pred)
    json.dump(res, open(a.baseline_out, "w"), indent=2)
    print("\nTEST baselines (persistence = today's score)")
    print(f"{'horizon/task':18s} {'n':>6s} {'MAE mean':>9s} {'MAE pers':>9s} {'AUROC pers':>10s} "
          f"{'onset rate':>10s} {'onset AUPRC':>11s}")
    for key, r in res.items():
        print(f"{key:18s} {r['n']:6d} {r['mae_mean']:9.3f} {r['mae_persistence']:9.3f} "
              f"{r['auroc_persistence']:10.3f} {r['onset_rate']:10.3f} {r['onset_auprc_persistence']:11.3f}")

    np.savez_compressed(
        a.out, X=Xz, E=Ez, S=S, H=Hz, score=np.nan_to_num(score).astype(np.float32),
        mask=mask, offset_days=np.nan_to_num(offset_days).astype(np.float32),
        current=cur, uid=UID, date=DATE.astype(str), split=split,
        horizons=np.array(HORIZONS_WEEKS), hist_names=np.array(HIST_NAMES),
        daily_names=np.array(DAILY + ["valid_mask"]), eng_names=np.array(engineered_names()),
        static_names=np.array(list(stat.columns) + ["ios_share"]),
        gender=demo["gender"].fillna("unknown").to_numpy().astype(str),
        race=demo["race"].fillna("unknown").to_numpy().astype(str))
    print("saved", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ema", required=True)
    p.add_argument("--sensing", required=True)
    p.add_argument("--demo", required=True)
    p.add_argument("--split_from", required=True, help="detection dataset, to reuse its student split")
    p.add_argument("--out", default="ces_forecast.npz")
    p.add_argument("--baseline_out", default="baselines_test.json")
    main(p.parse_args())
