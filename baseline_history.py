"""
History-only control (pre-registered in CLAUDE.md, 2026-10-22): simple reference baseline.

Per CV fold (same folds as train_forecast.py), horizon and task, fitted on [H, S] of the fold's
training students (the other 4 folds, 136 students):
    LinearRegression                          -> future score (clipped to 0..6 at evaluation)
    LogisticRegression(C=1.0, max_iter=1000)  -> screen-positive (score >= 3), logit output
Only rows with a target are used for fitting. Out-of-fold predictions cover the 170 train +
validation students once each; the test students are never used.

    python baseline_history.py --out runs/historyctl/baseline

RQ3 (pre-registered in CLAUDE.md): the same linear model and a linear quantile regression
(QuantileRegressor, alpha = 0, solver highs; 5 / 10 / 90 / 95% quantiles) on [H, S], fitted on the
model students of each nested-CV role (rq3_students; no early stopping, so all model students),
predicting the calibration and evaluation students (and the model students for the split role):
    python baseline_history.py --rq3 --outers 0 1 2 3 4 --out_root runs/rq3
"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from train_forecast import cv_folds, CUTOFF, RQ3_ROLES, rq3_rows
from quantile_model import QUANTILES, N_Q, sort_quantiles


def fit_predict(F_tr, score_tr, mask_tr, F_ho):
    """Returns (n_ho, n_horizons, 4) in the model layout: [dep score, anx score, dep logit, anx logit]."""
    n_h = score_tr.shape[1]
    out = np.zeros((len(F_ho), n_h, 4), np.float32)
    for k in range(n_h):
        m = mask_tr[:, k]
        for t in range(2):
            y = score_tr[m, k, t]
            out[:, k, t] = LinearRegression().fit(F_tr[m], y).predict(F_ho)
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(F_tr[m], (y >= CUTOFF).astype(int))
            out[:, k, 2 + t] = clf.decision_function(F_ho)
    return out


def fit_predict_quantile(F_tr, score_tr, mask_tr, F_out):
    """Linear quantile regression per horizon, task and quantile -> (n_out, n_horizons, 8) in the
    quantile-model layout (task-major, quantiles 5 / 10 / 90 / 95%), sorted so they never cross."""
    n_h = score_tr.shape[1]
    out = np.zeros((len(F_out), n_h, 2 * N_Q), np.float32)
    for k in range(n_h):
        m = mask_tr[:, k]
        for t in range(2):
            for i, q in enumerate(QUANTILES):
                qr = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(F_tr[m], score_tr[m, k, t])
                out[:, k, t * N_Q + i] = qr.predict(F_out)
    return sort_quantiles(out)


def rq3_main(a):
    d = np.load(a.data)
    uid, split = d["uid"], d["split"]
    F = np.concatenate([d["H"], d["S"]], 1).astype(np.float64)
    jobs = [("linear", k, r) for k in a.outers for r in a.roles]
    jobs += [("linearq", k, r) for k in a.outers for r in a.quantile_roles]
    for name, k, r in jobs:
        out_dir = os.path.join(a.out_root, f"{name}_H_outer{k}_{r}")
        if os.path.exists(os.path.join(out_dir, "results.json")) and not a.overwrite:
            print(f"skip {out_dir}"); continue
        parts = rq3_rows(uid, split, k, r)
        assert (split[np.concatenate(list(parts.values()))] < 2).all(), "a test-set row would be used"
        fit_rows = parts["model"]
        save = ["model", "cal", "eval"] if r == "split" else ["cal", "eval"]
        os.makedirs(out_dir, exist_ok=True)
        all_rows = np.concatenate([parts[p] for p in save])
        fn = fit_predict if name == "linear" else fit_predict_quantile
        pred_all = fn(F[fit_rows], d["score"][fit_rows], d["mask"][fit_rows], F[all_rows])   # fitted once
        start = 0
        for part in save:
            rows = parts[part]
            np.savez(os.path.join(out_dir, f"pred_{part}.npz"), rows=rows, pred=pred_all[start:start + len(rows)],
                     uid=uid[rows], date=d["date"][rows])
            start += len(rows)
        json.dump({"model": name, "version": "H", "outer": k, "role": r, "fit_rows": int(len(fit_rows)),
                   "fit_students": int(len(np.unique(uid[fit_rows])))},
                  open(os.path.join(out_dir, "results.json"), "w"), indent=2)
        print(f"{out_dir}: fitted on {len(fit_rows)} rows ({len(np.unique(uid[fit_rows]))} students)", flush=True)


def main(a):
    if a.rq3:
        return rq3_main(a)
    d = np.load(a.data)
    uid, split = d["uid"], d["split"]
    F = np.concatenate([d["H"], d["S"]], 1).astype(np.float64)
    tv = split < 2
    rows, preds, folds = [], [], []
    for f, held in enumerate(cv_folds(uid, split)):
        ho = np.where(tv & np.isin(uid, held))[0]
        tr = np.where(tv & ~np.isin(uid, held))[0]
        preds.append(fit_predict(F[tr], d["score"][tr], d["mask"][tr], F[ho]))
        rows.append(ho); folds.append(np.full(len(ho), f))
        print(f"fold {f}: fitted on {len(tr)} rows ({len(np.unique(uid[tr]))} students), "
              f"predicted {len(ho)} rows ({len(held)} students)")
    rows, preds, folds = np.concatenate(rows), np.concatenate(preds), np.concatenate(folds)
    order = np.argsort(rows)
    rows, preds, folds = rows[order], preds[order], folds[order]
    assert np.array_equal(rows, np.where(tv)[0]), "out-of-fold rows must cover train+val exactly once"
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "oof_predictions.npz"), rows=rows, pred=preds, fold=folds,
             score=d["score"][rows], mask=d["mask"][rows], current=d["current"][rows],
             uid=uid[rows], date=d["date"][rows])
    print(f"saved {a.out}/oof_predictions.npz ({len(rows)} rows, {len(np.unique(uid[rows]))} students)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ces_forecast.npz")
    p.add_argument("--out", default="runs/historyctl/baseline")
    p.add_argument("--rq3", action="store_true", help="RQ3 nested-CV linear and quantile-linear fits")
    p.add_argument("--outers", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--roles", nargs="+", choices=RQ3_ROLES, default=RQ3_ROLES)
    p.add_argument("--quantile_roles", nargs="*", choices=RQ3_ROLES, default=["split"])
    p.add_argument("--out_root", default="runs/rq3")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
