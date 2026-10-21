"""
Step 1 of the KAN-MTLNet reproduction on CES: align the paper's setup to our data.

Paper (All of Us)                      ->  This script (CES)
-------------------------------------------------------------------------------
one 60-day Fitbit window per person    ->  one 60-day window per (student, PHQ-4 survey)
                                           window = the 60 days BEFORE the survey day
clinical depression / anxiety labels   ->  PHQ-2 >= 3 (depression screen-positive)
                                           GAD-2 >= 3 (anxiety screen-positive)
daily Fitbit features                  ->  21 cross-platform daily phone features (audit)
missing data: median + indicators      ->  invalid days set to missing, train-median
                                           imputation, plus a daily "valid" mask channel
engineered features (CV over 3/7/14d)  ->  same idea on our key features
static covariates (demographics)       ->  gender, race group, iPhone share
subject-level 70/10/20 split           ->  student-level 70/10/20 split

Usage:
    python data_prep.py --ema general_ema.csv --sensing sensing.csv \
                        --demo demographics.csv --out ces_kan_dataset.npz
"""
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings (every choice the paper does not fix is written down here)
# ---------------------------------------------------------------------------
WINDOW = 60                 # days of history, as in the paper
MIN_VALID_LAST14 = 7        # keep a survey only if >= 7 valid days in the 14 days before it
QUALITY_MIN_HOURS = 12      # a day is "valid" if activity AND location have >= 12 h of data
CUTOFF = 3                  # PHQ-2 / GAD-2 screen-positive threshold
SEED = 42

# Cross-platform daily features (Android-only / iOS-only features excluded, see audit)
DAILY = [
    "act_still_ep_0", "act_in_vehicle_ep_0", "act_on_bike_ep_0",          # activity
    "loc_dist_ep_0", "loc_visit_num_ep_0", "loc_max_dis_from_campus_ep_0",  # mobility
    "loc_home_dur", "loc_self_dorm_dur", "loc_study_dur", "loc_social_dur",  # places
    "loc_food_dur", "loc_workout_dur", "loc_other_dorm_dur", "loc_leisure_dur",
    "unlock_num_ep_0", "unlock_duration_ep_0",                              # phone use
    "unlock_num_ep_1", "unlock_duration_ep_1",                              # night-time phone use
    "sleep_duration", "sleep_start", "sleep_end",                           # sleep
]
# heavy-tailed counts / durations / distances are log-transformed
LOG_FEATS = [f for f in DAILY if not f.startswith("sleep")]

# features used for the engineered "variability / regularity" summaries
KEY = ["act_still_ep_0", "loc_dist_ep_0", "loc_visit_num_ep_0",
       "unlock_duration_ep_0", "sleep_duration", "sleep_start", "loc_home_dur"]
WEEKEND_KEY = ["act_still_ep_0", "loc_dist_ep_0", "sleep_duration", "unlock_duration_ep_0"]


def load_labels(path):
    e = pd.read_csv(path, usecols=["uid", "day", "phq4-1", "phq4-2", "phq4-3", "phq4-4"])
    e = e.dropna(subset=["phq4-1", "phq4-2", "phq4-3", "phq4-4"]).copy()
    e["date"] = pd.to_datetime(e["day"].astype(str), format="%Y%m%d")
    # recompute subscores from the items (the provided total disagrees in 182 rows)
    e["anx_score"] = e["phq4-1"] + e["phq4-2"]          # GAD-2
    e["dep_score"] = e["phq4-3"] + e["phq4-4"]          # PHQ-2
    e["y_dep"] = (e["dep_score"] >= CUTOFF).astype(np.float32)
    e["y_anx"] = (e["anx_score"] >= CUTOFF).astype(np.float32)
    return e[["uid", "date", "dep_score", "anx_score", "y_dep", "y_anx"]]


def load_sensing(path):
    cols = ["uid", "day", "is_ios", "quality_activity", "quality_loc"] + DAILY
    dtypes = {c: "float32" for c in cols if c not in ("uid", "day")}
    dtypes["day"] = "int64"
    s = pd.read_csv(path, usecols=cols, dtype=dtypes)
    s["date"] = pd.to_datetime(s["day"].astype(str), format="%Y%m%d")
    s["valid"] = ((s.quality_activity >= QUALITY_MIN_HOURS) &
                  (s.quality_loc >= QUALITY_MIN_HOURS)).astype(np.float32)
    # invalid day -> every feature missing (zeros there mean "not measured")
    s.loc[s.valid == 0, DAILY] = np.nan
    # implausible sleep (audit: 2% of days > 14 h)
    s.loc[(s.sleep_duration < 1) | (s.sleep_duration > 16), "sleep_duration"] = np.nan
    return s


def static_table(demo_path, uids):
    d = pd.read_csv(demo_path)
    d = pd.DataFrame({"uid": uids}).merge(d, on="uid", how="left")
    g = d["gender"].fillna("unknown")
    r = d["race"].fillna("unknown").str.lower()
    race = np.select(
        [r.eq("white"), r.eq("asian"), r.eq("black"), r.eq("other/hispanic"), r.eq("unknown")],
        ["white", "asian", "black", "hispanic_other", "unknown"], "multiracial_native")
    out = pd.DataFrame({"uid": uids})
    for name, val in [("female", "F"), ("male", "M")]:
        out[f"gender_{name}"] = (g == val).astype(np.float32).values
    out["gender_other_or_unknown"] = (~g.isin(["F", "M"])).astype(np.float32).values
    for rg in ["white", "asian", "black", "hispanic_other", "multiracial_native", "unknown"]:
        out[f"race_{rg}"] = (race == rg).astype(np.float32)
    return out.set_index("uid"), d.set_index("uid")[["gender", "race"]]


def engineered(raw, valid, weekday):
    """Window-level summaries computed on untransformed daily values.
    raw: (60, F) array with NaN, valid: (60,), weekday: (60,) 0=Mon..6=Sun"""
    feats = []
    idx = {f: i for i, f in enumerate(DAILY)}
    with np.errstate(all="ignore"):
        for f in KEY:
            x = raw[:, idx[f]]
            feats.append(np.nanmean(x[-14:]))
            for w in (3, 7, 14):                       # coefficient of variation, as in paper
                seg = x[-w:]
                m = np.nanmean(seg)
                feats.append(np.nanstd(seg) / (abs(m) + 1e-6) if np.sum(~np.isnan(seg)) >= 2 else np.nan)
        wk = weekday >= 5
        for f in WEEKEND_KEY:                          # weekend vs weekday pattern
            x = raw[:, idx[f]]
            feats.append(np.nanmean(x[wk]) - np.nanmean(x[~wk]))
    feats.append(valid[-14:].mean())
    feats.append(valid.mean())
    return np.array(feats, dtype=np.float32)


def engineered_names():
    n = []
    for f in KEY:
        n += [f"{f}_mean14", f"{f}_cv3", f"{f}_cv7", f"{f}_cv14"]
    n += [f"{f}_weekend_minus_weekday" for f in WEEKEND_KEY]
    return n + ["valid_frac_14", "valid_frac_60"]


def main(a):
    labels = load_labels(a.ema)
    sens = load_sensing(a.sensing)

    # transform daily features once (log for heavy tails), keep untransformed copy too
    sens_raw = sens.copy()
    for f in LOG_FEATS:
        sens[f] = np.log1p(sens[f].clip(lower=0))

    rng = np.random.default_rng(SEED)
    X, M, E, S_plat, Y, SC, UID, DATE = [], [], [], [], [], [], [], []
    by_uid = {u: g.set_index("date").sort_index() for u, g in sens.groupby("uid")}
    by_uid_raw = {u: g.set_index("date").sort_index() for u, g in sens_raw.groupby("uid")}

    for u, lab in labels.groupby("uid"):
        if u not in by_uid:
            continue
        g, graw = by_uid[u], by_uid_raw[u]
        full_idx = pd.date_range(g.index.min() - pd.Timedelta(days=WINDOW), g.index.max())
        g = g.reindex(full_idx)
        graw = graw.reindex(full_idx)
        feat = g[DAILY].to_numpy(np.float32)
        featraw = graw[DAILY].to_numpy(np.float32)
        val = g["valid"].fillna(0).to_numpy(np.float32)
        ios = g["is_ios"].to_numpy(np.float32)
        pos = {d: i for i, d in enumerate(full_idx)}
        for r in lab.itertuples(index=False):
            end = pos.get(r.date)                      # index of the survey day
            if end is None or end - WINDOW < 0:
                continue
            sl = slice(end - WINDOW, end)              # days t-60 .. t-1 (survey day excluded)
            v = val[sl]
            if v[-14:].sum() < MIN_VALID_LAST14:
                continue
            X.append(feat[sl]); M.append(v)
            wd = np.array([d.weekday() for d in full_idx[sl]])
            E.append(engineered(featraw[sl], v, wd))
            S_plat.append(np.nanmean(ios[sl]) if np.any(~np.isnan(ios[sl])) else np.nan)
            Y.append([r.y_dep, r.y_anx]); SC.append([r.dep_score, r.anx_score])
            UID.append(u); DATE.append(r.date)

    X = np.stack(X); M = np.stack(M); E = np.stack(E)
    Y = np.array(Y, np.float32); SC = np.array(SC, np.float32)
    UID = np.array(UID); DATE = np.array(DATE, dtype="datetime64[D]")
    print(f"samples kept: {len(X)} from {len(set(UID))} students "
          f"(surveys total {len(labels)})")

    # ---- student-level split 70/10/20 ----
    students = np.array(sorted(set(UID)))
    rng.shuffle(students)
    n = len(students)
    tr_s = set(students[: int(0.7 * n)])
    va_s = set(students[int(0.7 * n): int(0.8 * n)])
    split = np.array([0 if u in tr_s else 1 if u in va_s else 2 for u in UID], np.int8)

    # ---- normalisation with TRAINING students only ----
    tr = split == 0
    xtr = X[tr].reshape(-1, X.shape[2])
    mu, sd = np.nanmean(xtr, 0), np.nanstd(xtr, 0) + 1e-6
    Xz = (X - mu) / sd
    med = np.nanmedian(((xtr - mu) / sd), 0)
    Xz = np.where(np.isnan(Xz), med, Xz)
    Xz = np.clip(Xz, -5, 5).astype(np.float32)
    Xz = np.concatenate([Xz, M[..., None]], axis=2)    # last channel = valid-day mask

    Etr = E[tr]
    emed = np.nanmedian(Etr, 0)
    E = np.where(np.isnan(E), emed, E)
    emu, esd = E[tr].mean(0), E[tr].std(0) + 1e-6
    Ez = np.clip((E - emu) / esd, -5, 5).astype(np.float32)

    stat, demo_raw = static_table(a.demo, UID)
    plat = np.array(S_plat, np.float32)
    plat = np.where(np.isnan(plat), np.nanmedian(plat[tr]), plat)
    S = np.concatenate([stat.to_numpy(np.float32), plat[:, None]], axis=1)
    s_names = list(stat.columns) + ["ios_share"]

    for name, k in [("train", 0), ("val", 1), ("test", 2)]:
        m = split == k
        print(f"{name:5s}: students {len(set(UID[m])):3d}  samples {m.sum():6d}  "
              f"dep+ {Y[m,0].mean()*100:5.1f}%  anx+ {Y[m,1].mean()*100:5.1f}%")

    np.savez_compressed(
        a.out, X=Xz, E=Ez, S=S.astype(np.float32), Y=Y, scores=SC, uid=UID,
        date=DATE.astype(str), split=split,
        daily_names=np.array(DAILY + ["valid_mask"]),
        eng_names=np.array(engineered_names()), static_names=np.array(s_names),
        gender=demo_raw["gender"].fillna("unknown").to_numpy().astype(str),
        race=demo_raw["race"].fillna("unknown").to_numpy().astype(str))
    print("saved", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ema", required=True)
    p.add_argument("--sensing", required=True)
    p.add_argument("--demo", required=True)
    p.add_argument("--out", default="ces_kan_dataset.npz")
    main(p.parse_args())
