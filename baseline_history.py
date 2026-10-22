"""
History-only control (pre-registered in CLAUDE.md, 2026-10-22): simple reference baseline.

Per CV fold (same folds as train_forecast.py), horizon and task, fitted on [H, S] of the fold's
training students (the other 4 folds, 136 students):
    LinearRegression                          -> future score (clipped to 0..6 at evaluation)
    LogisticRegression(C=1.0, max_iter=1000)  -> screen-positive (score >= 3), logit output
Only rows with a target are used for fitting. Out-of-fold predictions cover the 170 train +
validation students once each; the test students are never used.

    python baseline_history.py --out runs/historyctl/baseline
"""
import argparse, os
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from train_forecast import cv_folds, CUTOFF


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


def main(a):
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
    main(p.parse_args())
