"""
RQ1 evaluation: persistence vs KAN no-ODE vs KAN-ODE, per horizon (1, 2, 4, 8 weeks).

    python evaluate_forecast.py                                          # development: validation students
    python evaluate_forecast.py --split test --final --seeds 0 1 2 3 4   # ONCE, at the very end

Fixed in advance (agreed with the student before any results were seen):
  * primary metric: MAE of the future score, averaged over depression (PHQ-2) and anxiety (GAD-2)
  * seeds are combined by averaging each model's predictions over seeds
    (scores are averaged; screen-positive logits are turned into probabilities and averaged)
  * decision rule, per input version: "the ODE helps" only if the 95% paired student-bootstrap CI
    of  MAE(KAN-ODE) - MAE(no-ODE)  lies entirely below zero at >= 2 of the 4 horizons
Secondary (reported, not part of the decision): per-task MAE, AUROC / AUC-PR for screen-positive
(>= 3), onset AUROC / AUC-PR among students currently below 3, spread over seeds.

CIs: resample STUDENTS with replacement (default 1000 times). Every method is scored on the same
resample, so differences between methods are paired. Model scores are clipped to 0..6.
Persistence = "future score = today's score"; today's score is also its ranking for AUROC.

Test-set protection: --split test needs --final and every run (2 models x 2 versions x seeds)
finished. It then writes runs/TEST_SET_USED.json; a second test evaluation is refused.
Baselines: persistence (today's score) and mean score (training students' average future score).
"""
import argparse, datetime, json, os, sys
import numpy as np
import torch

TASKS = ["depression", "anxiety"]
CUTOFF = 3.0
SCORE_RANGE = (0.0, 6.0)
MODELS = ["noode", "ode"]
METHODS = ["mean", "persistence", "noode", "ode"]
LABEL = {"mean": "Mean score", "persistence": "Persistence", "noode": "KAN no-ODE", "ode": "KAN-ODE"}
PAIRS = [("ode", "noode"), ("ode", "persistence"), ("noode", "persistence"), ("ode", "mean"), ("noode", "mean")]
VERSION_NAME = {"A": "sensing only", "B": "sensing + symptom history known at t0"}


# ----------------------------------------------------------------------------- weighted metrics
# Bootstrap resamples are expressed as row weights (how often each row's student was drawn).
def w_mae(err, w):
    return float((w * err).sum() / w.sum()) if w.sum() > 0 else np.nan


def w_auroc(y, s, w):
    """AUROC with sample weights; tied scores count 1/2 (same as sklearn)."""
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    _, start = np.unique(s, return_index=True)
    wp, wn = np.add.reduceat(w * y, start), np.add.reduceat(w * (1 - y), start)
    P, N = wp.sum(), wn.sum()
    if P <= 0 or N <= 0:
        return np.nan
    neg_below = np.cumsum(wn) - wn
    return float((wp * (neg_below + 0.5 * wn)).sum() / (P * N))


def w_auprc(y, s, w):
    """Average precision with sample weights (same definition as sklearn)."""
    order = np.argsort(-s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]              # last row of each tied score
    tp, fp = np.cumsum(w * y)[last], np.cumsum(w * (1 - y))[last]
    if len(tp) == 0 or tp[-1] <= 0:
        return np.nan
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / tp[-1]
    return float((np.diff(np.r_[0.0, recall]) * precision).sum())


# ----------------------------------------------------------------------------- metrics for all methods
def compute(preds, truth, w, horizons):
    """preds: {method: (score (n,H,2), rank (n,H,2))}; truth: score/mask/current; w: (n,) weights."""
    score, mask, cur = truth["score"], truth["mask"], truth["current"]
    out = {}
    for meth, (sc, rk) in preds.items():
        r = {}
        for k, h in enumerate(horizons):
            m = mask[:, k]
            ww = w[m]
            maes = []
            for t, name in enumerate(TASKS):
                y = score[m, k, t]
                pos = (y >= CUTOFF).astype(float)
                s = rk[m, k, t]
                neg_now = cur[m, t] < CUTOFF
                mae = w_mae(np.abs(sc[m, k, t] - y), ww)
                r[f"{h}w_{name}_mae"] = mae
                r[f"{h}w_{name}_auroc"] = w_auroc(pos, s, ww)
                r[f"{h}w_{name}_auprc"] = w_auprc(pos, s, ww)
                r[f"{h}w_{name}_onset_auroc"] = w_auroc(pos[neg_now], s[neg_now], ww[neg_now])
                r[f"{h}w_{name}_onset_auprc"] = w_auprc(pos[neg_now], s[neg_now], ww[neg_now])
                maes.append(mae)
            r[f"{h}w_mae_avg"] = float(np.mean(maes))               # PRIMARY metric
        out[meth] = r
    return out


def counts(truth, horizons):
    score, mask, cur = truth["score"], truth["mask"], truth["current"]
    out = {}
    for k, h in enumerate(horizons):
        m = mask[:, k]
        out[f"{h}w"] = {"n": int(m.sum()), "students": int(len(np.unique(truth["uid"][m])))}
        for t, name in enumerate(TASKS):
            pos = score[m, k, t] >= CUTOFF
            neg_now = cur[m, t] < CUTOFF
            out[f"{h}w"][name] = {"prevalence": float(pos.mean()), "onset_n": int(neg_now.sum()),
                                  "onset_rate": float(pos[neg_now].mean())}
    return out


def method_preds(model_preds, current, n_h, train_mean):
    """Baselines + seed-averaged model predictions as (score, rank) pairs.
    mean = the training students' average future score (a constant: no-skill ranking)."""
    pers = np.repeat(current[:, None, :], n_h, axis=1)
    const = np.broadcast_to(np.asarray(train_mean, np.float32), pers.shape)
    out = {"mean": (const, const), "persistence": (pers, pers)}
    for m, P in model_preds.items():                               # P: (n_seeds, n, H, 4)
        sc = np.clip(P[..., :2].mean(0), *SCORE_RANGE)
        prob = (1.0 / (1.0 + np.exp(-P[..., 2:]))).mean(0)
        out[m] = (sc, prob)
    return out


def bootstrap(preds, truth, horizons, n_boot, seed):
    """Paired student bootstrap: point estimates, CIs per method, CIs of differences."""
    point = compute(preds, truth, np.ones(len(truth["uid"])), horizons)
    students, inv = np.unique(truth["uid"], return_inverse=True)
    rng = np.random.default_rng(seed)
    keys = list(point["persistence"])
    samples = {m: {k: [] for k in keys} for m in preds}
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(students), len(students)), minlength=len(students))
        res = compute(preds, truth, cnt[inv].astype(float), horizons)
        for m in preds:
            for k in keys:
                samples[m][k].append(res[m][k])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out = {"point": point,
           "ci": {m: {k: ci(samples[m][k]) for k in keys} for m in preds},
           "diff": {}}
    for a, b in PAIRS:
        if a in preds and b in preds:
            out["diff"][f"{a}-{b}"] = {
                k: {"point": point[a][k] - point[b][k],
                    "ci": ci(np.array(samples[a][k]) - np.array(samples[b][k]))} for k in keys}
    return out


def decision(res, horizons):
    """Pre-registered rule on the PRIMARY metric (MAE avg): ODE helps if the paired CI of
    ODE - noODE is entirely below 0 at >= 2 horizons."""
    d = res["diff"]["ode-noode"]
    better = [h for h in horizons if d[f"{h}w_mae_avg"]["ci"][1] < 0]
    worse = [h for h in horizons if d[f"{h}w_mae_avg"]["ci"][0] > 0]
    return {"ode_better_at_weeks": better, "ode_worse_at_weeks": worse, "ode_helps": len(better) >= 2}


# ----------------------------------------------------------------------------- loading runs
def run_dir(root, m, v, s):
    return os.path.join(root, f"{m}_{v}_seed{s}")


def finished(root, m, v, s):
    return os.path.exists(os.path.join(run_dir(root, m, v, s), "results.json"))


def found_seeds(root, v):
    """Seeds for which BOTH models of version v finished."""
    seeds = set()
    if os.path.isdir(root):
        for name in os.listdir(root):
            if name.startswith(f"ode_{v}_seed"):
                s = int(name.rsplit("seed", 1)[1])
                if all(finished(root, m, v, s) for m in MODELS):
                    seeds.add(s)
    return sorted(seeds)


def load_val(root, v, seeds):
    preds, truth = {m: [] for m in MODELS}, None
    for m in MODELS:
        for s in seeds:
            f = np.load(os.path.join(run_dir(root, m, v, s), "val_predictions.npz"))
            if truth is None:
                truth = {k: f[k] for k in ["score", "mask", "current", "uid", "date"]}
            assert np.array_equal(f["uid"], truth["uid"]) and np.array_equal(f["date"], truth["date"]), \
                f"{m}_{v}_seed{s}: validation rows differ from the other runs"
            preds[m].append(f["pred"])
    return {m: np.stack(p) for m, p in preds.items()}, truth


def load_test(root, v, seeds, data_path, device):
    """Predict the TEST students with every saved model (only reached with --final)."""
    from train_forecast import load_data, build_model, predict
    data = load_data(data_path, device)
    rows = np.where(data["np_split"] == 2)[0]
    truth = {"score": data["np_score"][rows], "mask": data["np_mask"][rows],
             "current": data["np_current"][rows], "uid": data["np_uid"][rows], "date": data["np_date"][rows]}
    preds = {m: [] for m in MODELS}
    for m in MODELS:
        for s in seeds:
            d = run_dir(root, m, v, s)
            residual = json.load(open(os.path.join(d, "results.json")))["settings"].get("residual", False)
            model = build_model(m, v, data).to(device)
            model.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=device))
            p = predict(model, data, rows, v, residual)
            np.savez(os.path.join(d, "test_predictions.npz"), pred=p, uid=truth["uid"], date=truth["date"])
            preds[m].append(p)
    return {m: np.stack(p) for m, p in preds.items()}, truth


def per_seed(model_preds, truth, horizons, train_mean):
    """Spread over seeds of the primary metric (each seed scored on its own, no bootstrap)."""
    out = {}
    for m, P in model_preds.items():
        vals = []
        for i in range(P.shape[0]):
            r = compute(method_preds({m: P[i:i + 1]}, truth["current"], len(horizons), train_mean), truth,
                        np.ones(len(truth["uid"])), horizons)[m]
            vals.append([r[f"{h}w_mae_avg"] for h in horizons])
        vals = np.array(vals)
        out[m] = {f"{h}w_mae_avg": {"mean": float(vals[:, k].mean()), "sd": float(vals[:, k].std(ddof=1))
                                    if len(vals) > 1 else 0.0, "per_seed": vals[:, k].tolist()}
                  for k, h in enumerate(horizons)}
    return out


# ----------------------------------------------------------------------------- report
def fmt(p, c, d=3):
    return f"{p:.{d}f} [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def report(all_res, split, horizons):
    L = []
    if split == "val":
        L += ["# RQ1 development results (VALIDATION students)", "",
              "> Development only. These numbers are used to check and tune the models. "
              "The decision rule is applied to the test set, once, at the end.", ""]
    else:
        L += ["# RQ1 results (TEST students, evaluated once)", ""]
    for v, R in all_res.items():
        res, cnt = R["bootstrap"], R["counts"]
        P, C, D = res["point"], res["ci"], res["diff"]
        L += [f"## Version {v}: {VERSION_NAME[v]}  (seeds {R['seeds']}, predictions averaged over seeds)", "",
              "**Primary metric: MAE of the future score, averaged over depression and anxiety "
              "(lower is better).** 95% CI from a student-level bootstrap.", "",
              "| Horizon | n (students) | Mean score | Persistence | KAN no-ODE | KAN-ODE | KAN-ODE − no-ODE |",
              "|---|---|---|---|---|---|---|"]
        for h in horizons:
            k = f"{h}w_mae_avg"
            L.append(f"| {h} week{'s' if h > 1 else ''} | {cnt[f'{h}w']['n']} ({cnt[f'{h}w']['students']}) | "
                     + " | ".join(fmt(P[m][k], C[m][k]) for m in METHODS)
                     + f" | {fmt(D['ode-noode'][k]['point'], D['ode-noode'][k]['ci'])} |")
        dec = R["decision"]
        L += ["", f"**Decision rule** (ODE helps if the ODE − no-ODE CI is entirely below 0 at ≥ 2 horizons): "
              f"better at weeks {dec['ode_better_at_weeks'] or 'none'}, worse at weeks "
              f"{dec['ode_worse_at_weeks'] or 'none'} → **ODE helps: {'YES' if dec['ode_helps'] else 'NO'}**"
              + (" (validation only, not the RQ1 answer)" if split == "val" else ""), "",
              "Each model vs the baselines (MAE avg, negative = model better than the baseline):", "",
              "| Horizon | no-ODE − persistence | KAN-ODE − persistence | no-ODE − mean | KAN-ODE − mean |",
              "|---|---|---|---|---|"]
        for h in horizons:
            k = f"{h}w_mae_avg"
            L.append(f"| {h}w | " + " | ".join(fmt(D[p][k]['point'], D[p][k]['ci']) for p in
                     ["noode-persistence", "ode-persistence", "noode-mean", "ode-mean"]) + " |")
        L += ["", "Secondary metrics (point estimate [95% CI]); classification = screen-positive (score ≥ 3), "
              "onset = currently < 3:", "",
              "| Horizon | Task | Method | MAE | AUROC | AUC-PR | Onset AUROC | Onset AUC-PR |",
              "|---|---|---|---|---|---|---|---|"]
        for h in horizons:
            for name in TASKS:
                c = cnt[f"{h}w"][name]
                for m in METHODS:
                    b = f"{h}w_{name}"
                    L.append(f"| {h}w | {name} (prev. {c['prevalence']:.1%}, onset {c['onset_rate']:.1%}) | "
                             f"{LABEL[m]} | " + " | ".join(
                                 fmt(P[m][f'{b}_{x}'], C[m][f'{b}_{x}'])
                                 for x in ["mae", "auroc", "auprc", "onset_auroc", "onset_auprc"]) + " |")
        if len(R["seeds"]) > 1:
            L += ["", "Spread over seeds (MAE avg, each seed on its own: mean ± SD):", "",
                  "| Horizon | KAN no-ODE | KAN-ODE |", "|---|---|---|"]
            for h in horizons:
                k = f"{h}w_mae_avg"
                L.append(f"| {h}w | " + " | ".join(
                    f"{R['per_seed'][m][k]['mean']:.3f} ± {R['per_seed'][m][k]['sd']:.3f}" for m in MODELS) + " |")
        L.append("")
    return "\n".join(L)


def check_persistence(res, baseline_path, horizons):
    """On the test set, persistence must reproduce baselines_test.json from forecast_prep.py."""
    if not os.path.exists(baseline_path):
        return "baselines file not found, persistence not cross-checked"
    base = json.load(open(baseline_path))
    P, M = res["point"]["persistence"], res["point"]["mean"]
    diffs = [abs(P[f"{h}w_{t}_{x}"] - base[f"{h}w_{t}"][f"{x}_persistence"])
             for h in horizons for t in TASKS for x in ["mae", "auroc", "auprc", "onset_auroc", "onset_auprc"]]
    diffs += [abs(M[f"{h}w_{t}_mae"] - base[f"{h}w_{t}"]["mae_mean"]) for h in horizons for t in TASKS]
    return f"persistence and mean baselines vs {baseline_path}: max difference {max(diffs):.2e}"


# ----------------------------------------------------------------------------- main
def main(a):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # Windows console: allow − ≥ →
    d = np.load(a.data)
    horizons = d["horizons"].tolist()
    tr = d["split"] == 0
    train_mean = [float(d["score"][tr][..., t][d["mask"][tr]].mean()) for t in range(2)]
    if a.split == "test":
        if not a.final:
            sys.exit("Refused: the test set is used ONCE. Add --final (and --seeds) when every setting is frozen.")
        lock = a.lock
        if os.path.exists(lock) and not a.force:
            sys.exit(f"Refused: the test set was already evaluated ({json.load(open(lock))['time']}). "
                     f"See {lock}.")
        if not a.seeds:
            sys.exit("Refused: --final needs the explicit list of --seeds.")
        missing = [run_dir(a.runs, m, v, s) for v in a.versions for m in MODELS for s in a.seeds
                   if not finished(a.runs, m, v, s)]
        if missing:
            sys.exit("Refused: unfinished runs:\n  " + "\n  ".join(missing))
        os.makedirs(os.path.dirname(os.path.abspath(lock)), exist_ok=True)
        json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "seeds": a.seeds,
                   "versions": a.versions, "runs": a.runs, "argv": sys.argv}, open(lock, "w"), indent=2)
        print(f"test set evaluation started; lock written to {lock}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_res = {}
    for v in a.versions:
        seeds = a.seeds or found_seeds(a.runs, v)
        if not seeds:
            print(f"version {v}: no finished ode+noode pair in {a.runs}, skipped")
            continue
        missing = [run_dir(a.runs, m, v, s) for m in MODELS for s in seeds if not finished(a.runs, m, v, s)]
        if missing:
            sys.exit("Unfinished runs:\n  " + "\n  ".join(missing))
        model_preds, truth = (load_val(a.runs, v, seeds) if a.split == "val"
                              else load_test(a.runs, v, seeds, a.data, device))
        preds = method_preds(model_preds, truth["current"], len(horizons), train_mean)
        print(f"version {v}: seeds {seeds}, {len(truth['uid'])} rows, "
              f"{len(np.unique(truth['uid']))} students, {a.n_boot} bootstrap resamples ...", flush=True)
        res = bootstrap(preds, truth, horizons, a.n_boot, a.boot_seed)
        all_res[v] = {"seeds": seeds, "counts": counts(truth, horizons), "bootstrap": res,
                      "decision": decision(res, horizons),
                      "per_seed": per_seed(model_preds, truth, horizons, train_mean)}
        if a.split == "test":
            all_res[v]["persistence_check"] = check_persistence(res, a.baselines, horizons)
            print("  " + all_res[v]["persistence_check"])
    if not all_res:
        sys.exit("nothing to evaluate")

    out = a.out or os.path.join(a.runs, f"eval_{a.split}")
    os.makedirs(out, exist_ok=True)
    text = report(all_res, a.split, horizons)
    open(os.path.join(out, "rq1_table.md"), "w", encoding="utf-8").write(text)
    json.dump({"split": a.split, "n_boot": a.n_boot, "boot_seed": a.boot_seed, "results": all_res},
              open(os.path.join(out, "rq1_results.json"), "w"), indent=2)
    print("\n" + text)
    print(f"saved {out}/rq1_table.md and rq1_results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/forecast")
    p.add_argument("--data", default="data/ces_forecast.npz")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--final", action="store_true", help="required for --split test")
    p.add_argument("--force", action="store_true", help="ignore an existing test lock (do not use)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default (val): all finished seeds")
    p.add_argument("--versions", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument("--baselines", default="baselines_test.json")
    p.add_argument("--lock", default="runs/TEST_SET_USED.json",
                   help="one lock for the whole project: the test set is evaluated once, whatever --runs is")
    p.add_argument("--out", default=None)
    main(p.parse_args())
