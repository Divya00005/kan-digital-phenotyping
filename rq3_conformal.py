"""
RQ3 evaluation (pre-registered in CLAUDE.md on 2026-10-22, amendments recorded before training).

Can conformal prediction give reliable prediction sets for future PHQ-2 (depression) and GAD-2
(anxiety) scores at 1 / 2 / 4 / 8 weeks, overall and within gender and race groups?

Nested student-level CV on the 170 train + validation students (test set never used): outer fold k
(34 students) = evaluation students; calibration and model students come from the other 136.
Point models: persistence, linear (history), KAN no-ODE H and B, KAN no-ODE Hn (no gender / race),
KAN-ODE H. Quantile models (CQR): linear quantile regression, quantile KAN H. Models are read by
name from their saved prediction files (runs/rq3/<run>/pred_{model,cal,eval}.npz).

Methods (conformal.py), alpha = 0.10 (main) and 0.20:
  B        MAIN. Student-level repeated subsampling (Dunn et al.), 200 subsamples, split model.
           Guarantee >= 1 - 2 alpha for a random cutoff of a new student.
  B1       single subsample (exact >= 1 - alpha), reference
  A        survey-level split conformal (Lei et al.), reference (no new-student guarantee)
  pCDF     pooled-CDF (Dunn et al. Method 1), asymptotic only
  BCV      B + CV+ over the 4 inner models: EMPIRICAL ONLY, not covered by either paper's theorem
  CQR      method B with the CQR score on the quantile models
  per-group calibration (gender: F, M; race: White, Asian) for method B and CQR; the others pooled
  joint    Bonferroni (method B at alpha / 4 per horizon); extra: max-normalised joint score

Coverage is averaged over students (primary; matches the guarantee) and over cutoffs. 95% CIs from
1,000 student-level bootstrap resamples (seed 0) with the sets held fixed; differences are paired.

    python rq3_conformal.py
"""
import argparse, csv, json, os, sys, time
import numpy as np
import conformal as cf
from train_forecast import rq3_rows
from quantile_model import quantile_index

TASKS = ["depression", "anxiety"]
ALPHAS = [0.10, 0.20]
B_SUB = 200
POINT = {"persistence": None, "linear": "linear_H_outer{k}_{role}", "kan_H": "noode_H_outer{k}_{role}_seed0",
         "kan_B": "noode_B_outer{k}_{role}_seed0", "kan_Hn": "noode_Hn_outer{k}_{role}_seed0",
         "kanode_H": "ode_H_outer{k}_{role}_seed0"}
QUANT = {"linear_q": "linearq_H_outer{k}_{role}", "kan_Hq": "noode_q_H_outer{k}_{role}_seed0"}
CVPLUS = ["persistence", "linear", "kan_H", "kan_B"]
QLEVELS = {0.10: (0.05, 0.95), 0.20: (0.10, 0.90)}
PARTITIONS = {"gender": ["F", "M"], "race": ["white", "asian"]}
SMALL = ["black", "hispanic_other", "multiracial_native", "race_unknown", "gender_both", "gender_unknown"]
LABEL = {"persistence": "Persistence", "linear": "Linear (history)", "kan_H": "KAN no-ODE H",
         "kan_B": "KAN no-ODE B", "kan_Hn": "KAN no-ODE Hn (no gender/race)", "kanode_H": "KAN-ODE H",
         "linear_q": "Linear quantile (CQR)", "kan_Hq": "Quantile KAN H (CQR)"}
MLABEL = {"B": "B: student-level repeated subsampling (main)", "B1": "B1: single subsample",
          "A": "A: survey-level split conformal", "pCDF": "pooled CDF", "BCV": "B + CV+ (empirical only)",
          "Bbonf": "B at alpha/4 (Bonferroni, per horizon)"}


def race_group(r):
    r = str(r).lower()
    return {"white": "white", "asian": "asian", "black": "black", "other/hispanic": "hispanic_other",
            "unknown": "race_unknown"}.get(r, "multiracial_native")


def gender_group(g):
    return {"F": "F", "M": "M", "both": "gender_both"}.get(str(g), "gender_unknown")


class Preds:
    """Loads saved predictions and checks that their rows are the expected ones."""
    def __init__(self, root, d):
        self.root, self.d = root, d

    def point(self, model, k, role, part, rows):
        if POINT[model] is None:                                   # persistence: today's score
            return np.repeat(self.d["current"][rows][:, None, :], 4, axis=1).astype(np.float64)
        return self._load(POINT[model], k, role, part, rows)[..., :2].astype(np.float64)

    def quant(self, model, k, role, part, rows):
        return self._load(QUANT[model], k, role, part, rows).astype(np.float64)

    def _load(self, pattern, k, role, part, rows):
        f = np.load(os.path.join(self.root, pattern.format(k=k, role=role), f"pred_{part}.npz"))
        assert np.array_equal(f["rows"], rows), f"{pattern} outer {k} {role} {part}: rows differ"
        return f["pred"]


class Store:
    """Per evaluation row (all outer folds concatenated): coverage (nan = no target / not in group),
    width and number of score values, for every (method, model, task, horizon, alpha, calibration)."""
    def __init__(self):
        self.data, self.trivial = {}, {}

    def add(self, key, k, n_rows, idx, cov, width, size):
        if key not in self.data:
            self.data[key] = {}
        if k not in self.data[key]:
            self.data[key][k] = np.full((3, n_rows), np.nan)
        self.data[key][k][:, idx] = np.vstack([cov, width, size])

    def flat(self, key, outers):
        return np.concatenate([self.data[key][k] for k in outers], axis=1)


def record(store, key, k, n_ev, idx, sets, y, has_target, width):
    cov = np.where(has_target, cf.covered(sets, np.nan_to_num(y)).astype(float), np.nan)
    size = np.where(has_target, cf.set_summary(sets)[2].astype(float), np.nan)
    store.add(key, k, n_ev, idx, cov, np.where(has_target, width, np.nan), size)


def rng(k):
    return np.random.default_rng(4000 + k)                          # fresh per combination (amendment)


def run_outer(k, d, P, store, models, quant_models, cvplus_models, groups_row):
    uid, score, mask = d["uid"], d["score"], d["mask"]
    sp = rq3_rows(uid, d["split"], k, "split")
    cal, ev, mod = sp["cal"], sp["eval"], sp["model"]
    n_ev = len(ev)
    all_ev = np.arange(n_ev)
    g_cal = {p: groups_row[p][cal] for p in PARTITIONS}
    g_ev = {p: groups_row[p][ev] for p in PARTITIONS}

    for m in models:
        Pc, Pe, Pm = (P.point(m, k, "split", part, rows) for part, rows in [("cal", cal), ("eval", ev), ("model", mod)])
        for t, task in enumerate(TASKS):
            for h in range(4):
                mc, me = mask[cal, h], mask[ev, h]
                s = cf.abs_score(score[cal, h, t], Pc[:, h, t])[mc]
                stu = uid[cal][mc]
                yhat, y = Pe[:, h, t], score[ev, h, t]
                cand = cf.abs_candidates(yhat)
                for a in ALPHAS:
                    thr = {"B": cf.subsample_threshold(s, stu, a, B_SUB, rng(k)),
                           "B1": cf.subsample_threshold(s, stu, a, 1, rng(k)),
                           "A": cf.split_threshold(s, a),
                           "pCDF": cf.pooled_cdf_threshold(s, stu, a),
                           "Bbonf": cf.subsample_threshold(s, stu, a / 4, B_SUB, rng(k))}
                    for meth, tt in thr.items():
                        record(store, (meth, m, t, h, a, "pooled"), k, n_ev, all_ev,
                               cf.sets_from_threshold(cand, tt), y, me, cf.abs_interval(yhat, tt)[2])
                    for part, wanted in PARTITIONS.items():          # per-group calibration, method B
                        gc = g_cal[part][mc]
                        for g in wanted:
                            tg = cf.subsample_threshold(s[gc == g], stu[gc == g], a, B_SUB, rng(k)) \
                                if (gc == g).any() else np.inf
                            store.trivial[("B", m, t, h, a, part, g, k)] = bool(np.isinf(tg))
                            idx = np.where(g_ev[part] == g)[0]
                            record(store, ("B", m, t, h, a, part), k, n_ev, idx,
                                   cf.sets_from_threshold(cand[idx], tg), y[idx], me[idx],
                                   cf.abs_interval(yhat[idx], tg)[2])
            # extra: max-normalised joint score (method B, pooled)
            scale = np.array([np.mean(np.abs(score[mod, h, t] - Pm[:, h, t])[mask[mod, h]]) for h in range(4)])
            mc_any = mask[cal].any(1)
            s_j = cf.max_norm_score(score[cal, :, t], Pc[:, :, t], mask[cal], scale)[mc_any]
            stu_j = uid[cal][mc_any]
            for a in ALPHAS:
                tt = cf.subsample_threshold(s_j, stu_j, a, B_SUB, rng(k))
                r = np.abs(score[ev, :, t] - Pe[:, :, t]) / scale[None, :]
                cov_h = r <= tt + cf.EPS
                joint = cf.joint_covered(cov_h, mask[ev])
                widths = np.stack([cf.abs_interval(Pe[:, h, t], tt * scale[h])[2] for h in range(4)], 1)
                w = np.where(mask[ev], widths, np.nan)
                has = mask[ev].any(1)
                wmean = np.where(has, np.nansum(w, 1) / np.maximum(mask[ev].sum(1), 1), np.nan)
                sz = np.stack([cf.set_summary(cf.sets_from_threshold(cf.abs_candidates(Pe[:, h, t]),
                                                                     tt * scale[h]))[2] for h in range(4)], 1)
                szmean = np.where(has, np.where(mask[ev], sz, 0).sum(1) / np.maximum(mask[ev].sum(1), 1), np.nan)
                store.add(("maxnorm", m, t, "joint", a, "pooled"), k, n_ev, all_ev, joint, wmean, szmean)

    for m in quant_models:                                           # CQR = method B with the CQR score
        Qc, Qe = P.quant(m, k, "split", "cal", cal), P.quant(m, k, "split", "eval", ev)
        for t, task in enumerate(TASKS):
            for h in range(4):
                mc, me = mask[cal, h], mask[ev, h]
                stu, y = uid[cal][mc], score[ev, h, t]
                for a in ALPHAS:
                    lo_q, hi_q = (quantile_index(t, q) for q in QLEVELS[a])
                    s = cf.cqr_score(score[cal, h, t], Qc[:, h, lo_q], Qc[:, h, hi_q])[mc]
                    cand = cf.cqr_candidates(Qe[:, h, lo_q], Qe[:, h, hi_q])
                    tt = cf.subsample_threshold(s, stu, a, B_SUB, rng(k))
                    record(store, ("B", m, t, h, a, "pooled"), k, n_ev, all_ev, cf.sets_from_threshold(cand, tt),
                           y, me, cf.cqr_interval(Qe[:, h, lo_q], Qe[:, h, hi_q], tt)[2])
                    for part, wanted in PARTITIONS.items():
                        gc = g_cal[part][mc]
                        for g in wanted:
                            tg = cf.subsample_threshold(s[gc == g], stu[gc == g], a, B_SUB, rng(k)) \
                                if (gc == g).any() else np.inf
                            store.trivial[("B", m, t, h, a, part, g, k)] = bool(np.isinf(tg))
                            idx = np.where(g_ev[part] == g)[0]
                            record(store, ("B", m, t, h, a, part), k, n_ev, idx,
                                   cf.sets_from_threshold(cand[idx], tg), y[idx], me[idx],
                                   cf.cqr_interval(Qe[idx, h, lo_q], Qe[idx, h, hi_q], tg)[2])

    for m in cvplus_models:                                          # B + CV+ (empirical only)
        inner = [rq3_rows(uid, d["split"], k, f"inner{j}") for j in range(4)]
        assert all(np.array_equal(r["eval"], ev) for r in inner)
        Pci = [P.point(m, k, f"inner{j}", "cal", inner[j]["cal"]) for j in range(4)]
        Pei = np.stack([P.point(m, k, f"inner{j}", "eval", ev) for j in range(4)], axis=-1)   # (n_ev, 4, 2, K)
        cal_rows = np.concatenate([inner[j]["cal"] for j in range(4)])
        fold = np.concatenate([np.full(len(inner[j]["cal"]), j) for j in range(4)])
        for t, task in enumerate(TASKS):
            for h in range(4):
                pred_cal = np.concatenate([Pci[j][:, h, t] for j in range(4)])
                mc = mask[cal_rows, h]
                resid = np.abs(score[cal_rows, h, t] - pred_cal)[mc]
                for a in ALPHAS:
                    sets, width = cf.cvplus_subsample_sets(Pei[:, h, t, :], fold[mc], resid, uid[cal_rows][mc],
                                                           a, B_SUB, rng(k), return_width=True)
                    record(store, ("BCV", m, t, h, a, "pooled"), k, n_ev, all_ev, sets, score[ev, h, t],
                           mask[ev, h], width)
    return ev


# ----------------------------------------------------------------------------- statistics
def student_table(arr, stu_idx, n_stu):
    """arr (3, n_rows) [cov, width, size] -> per-student sums: n, cov, width, size (nan rows ignored)."""
    ok = ~np.isnan(arr[0])
    n = np.bincount(stu_idx[ok], minlength=n_stu).astype(float)
    sums = [np.bincount(stu_idx[ok], weights=arr[i][ok], minlength=n_stu) for i in range(3)]
    return n, sums[0], sums[1], sums[2]


def stats(W, n, c, w, z, sel):
    """Bootstrap rows of W (first row = point estimate). sel: students in the group."""
    has = (n > 0) & sel
    Wh = W * has[None, :]
    den = Wh.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov_s = (Wh * np.where(has, c / np.maximum(n, 1), 0)).sum(1) / den
        cov_c = (Wh * c).sum(1) / (Wh * n).sum(1)
        wid = (Wh * np.where(has, w / np.maximum(n, 1), 0)).sum(1) / den
        siz = (Wh * np.where(has, z / np.maximum(n, 1), 0)).sum(1) / den
    return {"cov": cov_s, "cov_cut": cov_c, "width": wid, "size": siz,
            "n_students": int(has.sum()), "n_cutoffs": int(n[has].sum())}


def ci(v):
    v = v[1:][~np.isnan(v[1:])]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def main(a):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t_start = time.time()
    raw = np.load(a.data)
    d = {k: raw[k] for k in ["uid", "split", "score", "mask", "current", "gender", "race", "date"]}
    groups_row = {"gender": np.array([gender_group(g) for g in d["gender"]]),
                  "race": np.array([race_group(r) for r in d["race"]])}
    P, store = Preds(a.runs, d), Store()
    ev_rows = []
    for k in a.outers:
        print(f"outer fold {k} ... ({(time.time() - t_start) / 60:.1f} min)", flush=True)
        ev_rows.append(run_outer(k, d, P, store, a.models, a.quant_models, a.cvplus_models, groups_row))
    ev_rows = np.concatenate(ev_rows)
    assert (d["split"][ev_rows] < 2).all(), "a test-set row was evaluated"
    assert len(np.unique(ev_rows)) == len(ev_rows)

    uid_ev = d["uid"][ev_rows]
    students, stu_idx = np.unique(uid_ev, return_inverse=True)
    n_stu = len(students)
    first = {s: i for i, s in reversed(list(enumerate(uid_ev)))}
    g_stu = {p: groups_row[p][ev_rows][[first[s] for s in students]] for p in PARTITIONS}
    sel = {"all": np.ones(n_stu, bool)}
    for p in PARTITIONS:
        for g in np.unique(g_stu[p]):
            sel[g] = g_stu[p] == g
    all4 = d["mask"][ev_rows].all(1)

    br = np.random.default_rng(a.boot_seed)
    W = np.ones((a.n_boot + 1, n_stu))
    for b in range(1, a.n_boot + 1):
        W[b] = np.bincount(br.integers(0, n_stu, n_stu), minlength=n_stu)

    rows, tables, cache = [], {}, {}
    for key in store.data:
        meth, m, t, h, alpha, calib = key
        arr = store.flat(key, a.outers)
        cache[key] = student_table(arr, stu_idx, n_stu)
        if h == "joint" or meth == "Bbonf":
            continue
        groups = ["all"] + [g for g in sel if g != "all"] if calib == "pooled" else PARTITIONS[calib]
        for g in groups:
            st = stats(W, *cache[key], sel[g])
            rows.append(dict(method=meth, model=m, task=TASKS[t], horizon=[1, 2, 4, 8][h], alpha=alpha,
                             calibration=calib, group=g, n_students=st["n_students"], n_cutoffs=st["n_cutoffs"],
                             coverage=st["cov"][0], cov_lo=ci(st["cov"])[0], cov_hi=ci(st["cov"])[1],
                             coverage_cutoffs=st["cov_cut"][0], cov_cut_lo=ci(st["cov_cut"])[0],
                             cov_cut_hi=ci(st["cov_cut"])[1], cov_minus_target=st["cov"][0] - (1 - alpha),
                             width=st["width"][0], width_lo=ci(st["width"])[0], width_hi=ci(st["width"])[1],
                             values=st["size"][0], values_lo=ci(st["size"])[0], values_hi=ci(st["size"])[1],
                             under_coverage=bool(ci(st["cov"])[1] < 1 - alpha)))
    R = {(r["method"], r["model"], r["task"], r["horizon"], r["alpha"], r["calibration"], r["group"]): r for r in rows}

    # joint coverage: Bonferroni from per-horizon B at alpha / 4, and the max-normalised extra
    joint_rows = []
    for m in a.models:
        for t in range(2):
            for alpha in ALPHAS:
                per_h = [store.flat(("Bbonf", m, t, h, alpha, "pooled"), a.outers) for h in range(4)]
                cov_h = np.stack([x[0] for x in per_h], 1)
                jc = cf.joint_covered(cov_h == 1, ~np.isnan(cov_h))
                wid = np.nanmean(np.stack([x[1] for x in per_h], 1), 1)
                siz = np.nanmean(np.stack([x[2] for x in per_h], 1), 1)
                for name, arr in [("Bonferroni", np.vstack([jc, wid, siz])),
                                  ("max-normalised (extra)", store.flat(("maxnorm", m, t, "joint", alpha, "pooled"),
                                                                       a.outers))]:
                    for subset, keep in [("available horizons", np.ones(len(ev_rows), bool)), ("all 4 horizons", all4)]:
                        x = arr.copy(); x[:, ~keep] = np.nan
                        st = stats(W, *student_table(x, stu_idx, n_stu), sel["all"])
                        joint_rows.append(dict(model=m, task=TASKS[t], alpha=alpha, method=name, cutoffs=subset,
                                               n_cutoffs=st["n_cutoffs"], coverage=st["cov"][0], ci=ci(st["cov"]),
                                               width=st["width"][0], values=st["size"][0],
                                               under_coverage=bool(ci(st["cov"])[1] < 1 - alpha)))

    # gaps (paired) and model comparisons (paired), method B
    gap_rows, comp_rows = [], []
    for m in a.models + a.quant_models:
        for calib in ["pooled", "gender", "race"]:
            for p, (g1, g2) in [("gender", ("F", "M")), ("race", ("white", "asian"))]:
                if calib not in ("pooled", p):
                    continue
                for t in range(2):
                    for h in range(4):
                        for alpha in ALPHAS:
                            key = ("B", m, t, h, alpha, calib)
                            if key not in cache:
                                continue
                            s1, s2 = stats(W, *cache[key], sel[g1]), stats(W, *cache[key], sel[g2])
                            gap = s1["cov"] - s2["cov"]
                            c = ci(gap)
                            gap_rows.append(dict(model=m, calibration=calib, partition=p, task=TASKS[t],
                                                 horizon=[1, 2, 4, 8][h], alpha=alpha, gap=f"{g1} - {g2}",
                                                 cov_1=s1["cov"][0], cov_2=s2["cov"][0], diff=gap[0], ci=c,
                                                 gap_flag=bool(c[0] > 0 or c[1] < 0)))
    pairs = [("kan_H", "persistence"), ("kan_B", "kan_H"), ("kan_H", "linear"), ("kan_Hn", "kan_H"),
             ("kanode_H", "kan_H"), ("kan_Hq", "kan_H"), ("linear_q", "linear")]
    for m1, m2 in pairs:
        for t in range(2):
            for h in range(4):
                for alpha in ALPHAS:
                    k1, k2 = ("B", m1, t, h, alpha, "pooled"), ("B", m2, t, h, alpha, "pooled")
                    if k1 not in cache or k2 not in cache:
                        continue
                    s1, s2 = stats(W, *cache[k1], sel["all"]), stats(W, *cache[k2], sel["all"])
                    comp_rows.append(dict(model_1=m1, model_2=m2, task=TASKS[t], horizon=[1, 2, 4, 8][h],
                                          alpha=alpha, d_values=(s1["size"] - s2["size"])[0],
                                          d_values_ci=ci(s1["size"] - s2["size"]),
                                          d_width=(s1["width"] - s2["width"])[0], d_width_ci=ci(s1["width"] - s2["width"]),
                                          d_cov=(s1["cov"] - s2["cov"])[0], d_cov_ci=ci(s1["cov"] - s2["cov"])))

    # pre-registered verdict per model: method B, no under-coverage overall (pooled) and none within
    # F, M, White, Asian under per-group calibration (both targets, all tasks and horizons)
    verdict = {}
    for m in a.models + a.quant_models:
        over = [r for r in rows if r["method"] == "B" and r["model"] == m and r["calibration"] == "pooled"
                and r["group"] == "all"]
        grp = [r for r in rows if r["method"] == "B" and r["model"] == m and r["calibration"] != "pooled"]
        n_over, n_grp = sum(r["under_coverage"] for r in over), sum(r["under_coverage"] for r in grp)
        verdict[m] = dict(reliable=(n_over == 0 and n_grp == 0), under_overall=n_over, checks_overall=len(over),
                          under_within_groups=n_grp, checks_groups=len(grp),
                          mean_over_coverage_90=float(np.mean([r["cov_minus_target"] for r in over if r["alpha"] == 0.1])),
                          trivial_group_sets=sum(v for kk, v in store.trivial.items() if kk[1] == m))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "all_results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump({"verdict": verdict, "joint": joint_rows, "gaps": gap_rows, "comparisons": comp_rows,
               "n_students": n_stu, "n_cutoffs": int(len(ev_rows)), "outers": a.outers, "n_boot": a.n_boot,
               "B": B_SUB, "group_counts": {g: int(s.sum()) for g, s in sel.items()}},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    text = report(R, joint_rows, gap_rows, comp_rows, verdict, sel, a)
    open(os.path.join(a.out, "rq3_table.md"), "w", encoding="utf-8").write(text)
    print(text)
    print(f"\nsaved {a.out}/rq3_table.md, all_results.csv, summary.json "
          f"({(time.time() - t_start) / 60:.1f} min)")


# ----------------------------------------------------------------------------- report
def f3(x, lo=None, hi=None):
    return f"{x:.3f}" if lo is None else f"{x:.3f} [{lo:.3f}, {hi:.3f}]"


def report(R, joint_rows, gap_rows, comp_rows, verdict, sel, a):
    H = [1, 2, 4, 8]
    L = ["# RQ3: conformal prediction sets (nested student-level CV, 170 train + validation students)", "",
         "Test set not used. Coverage = share of forecasts whose true score lies in the set, averaged over "
         f"students (each student counts once; this is what the guarantee is about). 95% CIs: {a.n_boot:,} "
         "student-level bootstrap resamples, sets held fixed. 'values' = mean number of the 7 possible "
         "scores (0-6) in the set. Over-coverage is expected with integer scores (no randomised tie-breaking).",
         "", f"Students per group: {', '.join(f'{g} {int(s.sum())}' for g, s in sel.items())}", ""]
    models = [m for m in a.models + a.quant_models]

    def main_table(alpha):
        L.extend([f"## Method B (main), pooled calibration, target {int(100 * (1 - alpha))}%", "",
                  "| Model | Task | " + " | ".join(f"{h} wk: coverage [CI] · values" for h in H) + " |",
                  "|---|---|" + "---|" * 4])
        for m in models:
            for task in TASKS:
                cells = []
                for h in H:
                    r = R.get(("B", m, task, h, alpha, "pooled", "all"))
                    cells.append("–" if r is None else f"{f3(r['coverage'], r['cov_lo'], r['cov_hi'])} · {r['values']:.2f}")
                L.append(f"| {LABEL[m]} | {task} | " + " | ".join(cells) + " |")
        L.append("")
    main_table(0.10)
    main_table(0.20)

    L += ["## Methods compared, target 90%, pooled calibration (coverage [CI] · values)", "",
          "| Model | Method | Task | " + " | ".join(f"{h} wk" for h in H) + " |", "|---|---|---|" + "---|" * 4]
    for m in a.cvplus_models + [x for x in a.models if x not in a.cvplus_models]:
        for meth in ["B", "B1", "A", "pCDF", "BCV"]:
            for task in TASKS:
                cells = [R.get((meth, m, task, h, 0.10, "pooled", "all")) for h in H]
                if all(c is None for c in cells):
                    continue
                L.append(f"| {LABEL[m]} | {MLABEL[meth]} | {task} | " + " | ".join(
                    "–" if c is None else f"{f3(c['coverage'], c['cov_lo'], c['cov_hi'])} · {c['values']:.2f}"
                    for c in cells) + " |")
    L += ["", "B + CV+ is an empirical comparison only: the combination of student subsampling and CV+ is not "
          "covered by either paper's theorem. CV+'s own bound for one subsample with n = 136 students and "
          "K = 4 is 1 - 2 alpha - 0.043 (0.757 at 90%, 0.557 at 80%).", ""]

    for part, (g1, g2) in [("gender", ("F", "M")), ("race", ("white", "asian"))]:
        L += [f"## Coverage by {part}, method B, target 90% (pooled vs per-group calibration)", "",
              "| Model | Calibration | Task | " + " | ".join(f"{h} wk: {g1} / {g2} · gap [CI]" for h in H) + " |",
              "|---|---|---|" + "---|" * 4]
        for m in models:
            for calib in ["pooled", part]:
                for task in TASKS:
                    cells = []
                    for h in H:
                        gr = [x for x in gap_rows if x["model"] == m and x["calibration"] == calib and
                              x["partition"] == part and x["task"] == task and x["horizon"] == h and x["alpha"] == 0.10]
                        if not gr:
                            cells.append("–"); continue
                        x = gr[0]
                        flag = " ⚑" if x["gap_flag"] else ""
                        cells.append(f"{x['cov_1']:.3f} / {x['cov_2']:.3f} · {x['diff']:+.3f} "
                                     f"[{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]{flag}")
                    L.append(f"| {LABEL[m]} | {'per-group' if calib != 'pooled' else 'pooled'} | {task} | "
                             + " | ".join(cells) + " |")
        L.append("")
    L += ["⚑ = group-gap flag (paired CI excludes zero). Per-group calibration guarantees coverage within each "
          "group; pooled calibration does not.", "",
          "## Small groups (pooled calibration, method B, target 90%): too small to conclude", "",
          "| Model | Group | Students | Task | " + " | ".join(f"{h} wk" for h in H) + " |", "|---|---|---|---|" + "---|" * 4]
    for m in ["persistence", "kan_H"]:
        for g in SMALL:
            for task in TASKS:
                cells = [R.get(("B", m, task, h, 0.10, "pooled", g)) for h in H]
                if all(c is None for c in cells):
                    continue
                n = next(c for c in cells if c is not None)["n_students"]
                L.append(f"| {LABEL[m]} | {g} | {n} | {task} | " + " | ".join(
                    "–" if c is None else f"{c['coverage']:.2f}" for c in cells) + " |")

    L += ["", "## Joint coverage over horizons (method B, pooled)", "",
          "| Model | Task | Target | Method | Cutoffs | Joint coverage [CI] | Mean values per horizon |",
          "|---|---|---|---|---|---|---|"]
    for r in joint_rows:
        L.append(f"| {LABEL[r['model']]} | {r['task']} | {int(100 * (1 - r['alpha']))}% | {r['method']} | "
                 f"{r['cutoffs']} | {f3(r['coverage'], *r['ci'])} | {r['values']:.2f} |")

    L += ["", "## Model comparisons, method B, pooled, target 90% (paired; negative values = model 1 narrower)", "",
          "| Model 1 − model 2 | Task | " + " | ".join(f"{h} wk: Δ values [CI] · Δ coverage" for h in H) + " |",
          "|---|---|" + "---|" * 4]
    for m1, m2 in dict.fromkeys((c["model_1"], c["model_2"]) for c in comp_rows):
        for task in TASKS:
            cells = []
            for h in H:
                c = [x for x in comp_rows if x["model_1"] == m1 and x["model_2"] == m2 and x["task"] == task
                     and x["horizon"] == h and x["alpha"] == 0.10]
                cells.append("–" if not c else f"{c[0]['d_values']:+.2f} [{c[0]['d_values_ci'][0]:+.2f}, "
                             f"{c[0]['d_values_ci'][1]:+.2f}] · {c[0]['d_cov']:+.3f}")
            L.append(f"| {LABEL[m1]} − {LABEL[m2]} | {task} | " + " | ".join(cells) + " |")

    L += ["", "## Pre-registered verdict (method B)", "",
          "Reliable = no under-coverage flag (coverage CI entirely below target) overall with pooled "
          "calibration, and none within female, male, White, Asian with per-group calibration; both targets, "
          "both tasks, all horizons.", "",
          "| Model | Reliable | Under-coverage overall | Under-coverage within groups | Mean coverage − 90% | "
          "Trivial group sets |", "|---|---|---|---|---|---|"]
    for m, v in verdict.items():
        L.append(f"| {LABEL[m]} | **{'YES' if v['reliable'] else 'NO'}** | {v['under_overall']} of "
                 f"{v['checks_overall']} | {v['under_within_groups']} of {v['checks_groups']} | "
                 f"{v['mean_over_coverage_90']:+.3f} | {v['trivial_group_sets']} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ces_forecast.npz")
    p.add_argument("--runs", default="runs/rq3")
    p.add_argument("--out", default="runs/rq3/eval")
    p.add_argument("--outers", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--models", nargs="+", default=list(POINT))
    p.add_argument("--quant_models", nargs="*", default=list(QUANT))
    p.add_argument("--cvplus_models", nargs="*", default=CVPLUS)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--boot_seed", type=int, default=0)
    main(p.parse_args())
