"""
History-only control (pre-registered in CLAUDE.md, 2026-10-22): pooled cross-validation table.

Does version B's ranking gain over persistence come from phone sensing (B) or only from the
symptom-history features (H = same model, sensing inputs set to zero)?

  * out-of-fold predictions of the 170 train + validation students (test set never used)
  * per fold and version: predictions averaged over the 5 seeds, then all folds pooled
  * rows: mean score (training-fold mean), persistence, simple linear/logistic baseline on
    history + static, KAN version H, KAN version B
  * primary: screen-positive AUROC averaged over depression and anxiety, per horizon
  * rule: "sensing adds ranking information" only if the paired 95% student-bootstrap CI of
    AUROC(B) - AUROC(H) is entirely above zero at >= 2 of the 4 horizons

    python evaluate_history_control.py
"""
import argparse, json, os, sys
import numpy as np
import evaluate_forecast as ev

LABEL = {"mean": "Mean score", "persistence": "Persistence", "linbase": "Linear/logistic on history",
         "H": "KAN version H (history only)", "B": "KAN version B (sensing + history)"}
METHODS = ["mean", "persistence", "linbase", "H", "B"]
PAIRS = [("B", "H"), ("B", "persistence"), ("H", "persistence"), ("linbase", "persistence"),
         ("B", "linbase"), ("H", "linbase")]
FOLDS, SEEDS = range(5), range(5)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def as_method(P):
    """Raw outputs (n, H, 4) -> (clipped score, probability) as used by evaluate_forecast."""
    return np.clip(P[..., :2], *ev.SCORE_RANGE), sigmoid(P[..., 2:])


def load_version(root, v, rows_all, seeds=SEEDS):
    """Per fold: average scores and probabilities over seeds; pool folds into train+val row order."""
    sc = np.full((len(rows_all), 4, 2), np.nan, np.float64)
    pr = np.full((len(rows_all), 4, 2), np.nan, np.float64)
    pos = {r: i for i, r in enumerate(rows_all)}
    for f in FOLDS:
        runs = [np.load(os.path.join(root, f"noode_{v}_fold{f}_seed{s}", "oof_predictions.npz")) for s in seeds]
        rows = runs[0]["rows"]
        assert all(np.array_equal(r["rows"], rows) for r in runs)
        P = np.stack([r["pred"] for r in runs])                              # (seeds, n, H, 4)
        idx = [pos[r] for r in rows]
        sc[idx] = np.clip(P[..., :2].mean(0), *ev.SCORE_RANGE)
        pr[idx] = sigmoid(P[..., 2:]).mean(0)
    assert not np.isnan(sc).any(), f"version {v}: some rows have no out-of-fold prediction"
    return sc, pr


def main(a):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = np.load(a.data)
    split, uid = d["split"], d["uid"]
    horizons = d["horizons"].tolist()
    rows = np.where(split < 2)[0]                                            # train + val only
    truth = {"score": d["score"][rows], "mask": d["mask"][rows], "current": d["current"][rows],
             "uid": uid[rows]}

    # mean score: the training-fold mean of each task (rows with a target), per held-out fold
    from train_forecast import cv_folds
    fold_of = np.full(len(rows), -1)
    mean_sc = np.zeros((len(rows), 4, 2))
    for f, held in enumerate(cv_folds(uid, split)):
        ho = np.isin(uid[rows], held)
        tr = rows[~ho]
        fold_of[ho] = f
        for t in range(2):
            mean_sc[ho, :, t] = d["score"][tr][..., t][d["mask"][tr]].mean()
    assert (fold_of >= 0).all()

    base = np.load(os.path.join(a.runs, "baseline", "oof_predictions.npz"))
    assert np.array_equal(base["rows"], rows)
    pers = np.repeat(truth["current"][:, None, :], len(horizons), axis=1).astype(np.float64)
    preds = {"mean": (mean_sc, mean_sc), "persistence": (pers, pers), "linbase": as_method(base["pred"]),
             "H": load_version(a.runs, "H", rows), "B": load_version(a.runs, "B", rows)}

    ev.PAIRS = PAIRS
    print(f"{len(rows)} rows, {len(np.unique(truth['uid']))} students, {a.n_boot} bootstrap resamples ...",
          flush=True)
    res = ev.bootstrap(preds, truth, horizons, a.n_boot, a.boot_seed)
    P, C, D = res["point"], res["ci"], res["diff"]

    d_bh = D["B-H"]
    better = [h for h in horizons if d_bh[f"{h}w_auroc_avg"]["ci"][0] > 0]
    worse = [h for h in horizons if d_bh[f"{h}w_auroc_avg"]["ci"][1] < 0]
    decision = {"B_better_at_weeks": better, "B_worse_at_weeks": worse,
                "sensing_adds_ranking_information": len(better) >= 2}

    # spread over seeds (pooled over folds, one seed at a time) and over folds (seed-averaged)
    spread = {}
    for v in ["H", "B"]:
        per_seed = []
        for s in SEEDS:
            r = ev.compute({v: load_version(a.runs, v, rows, seeds=[s])}, truth, np.ones(len(rows)), horizons)[v]
            per_seed.append([r[f"{h}w_auroc_avg"] for h in horizons])
        per_fold = []
        for f in FOLDS:
            m = fold_of == f
            sub = {k: x[m] for k, x in truth.items()}
            r = ev.compute({v: tuple(x[m] for x in preds[v])}, sub, np.ones(m.sum()), horizons)[v]
            per_fold.append([r[f"{h}w_auroc_avg"] for h in horizons])
        spread[v] = {"per_seed": np.array(per_seed).tolist(), "per_fold": np.array(per_fold).tolist()}

    fmt = ev.fmt
    L = ["# History-only control: pooled cross-validation (170 train + validation students)", "",
         "Out-of-fold predictions, 5 folds x 5 seeds; test set not used. 95% CI from a student-level "
         "bootstrap (1,000 resamples); differences are paired.", "",
         "**Primary: screen-positive AUROC, averaged over depression and anxiety (higher is better).**", "",
         "| Horizon | " + " | ".join(LABEL[m] for m in METHODS) + " | B − H |",
         "|---|" + "---|" * (len(METHODS) + 1)]
    for h in horizons:
        k = f"{h}w_auroc_avg"
        L.append(f"| {h}w | " + " | ".join(fmt(P[m][k], C[m][k]) for m in METHODS)
                 + f" | {fmt(d_bh[k]['point'], d_bh[k]['ci'])} |")
    L += ["", f"**Decision rule** (sensing adds ranking information if the B − H CI is entirely above 0 at "
          f"≥ 2 horizons): B better at weeks {better or 'none'}, B worse at weeks {worse or 'none'} → "
          f"**sensing adds ranking information: {'YES' if decision['sensing_adds_ranking_information'] else 'NO'}**", ""]
    for title, key, better_txt in [("Onset AUC-PR, averaged over tasks (higher is better)", "onset_auprc_avg", ""),
                                   ("AUC-PR, averaged over tasks (higher is better)", "auprc_avg", ""),
                                   ("MAE, averaged over tasks (lower is better)", "mae_avg", "")]:
        L += [f"**{title}**", "", "| Horizon | " + " | ".join(LABEL[m] for m in METHODS) + " | B − H |",
              "|---|" + "---|" * (len(METHODS) + 1)]
        for h in horizons:
            k = f"{h}w_{key}"
            L.append(f"| {h}w | " + " | ".join(fmt(P[m][k], C[m][k]) for m in METHODS)
                     + f" | {fmt(d_bh[k]['point'], d_bh[k]['ci'])} |")
        L.append("")
    L += ["**Paired differences in AUROC (avg over tasks)**", "",
          "| Horizon | " + " | ".join(f"{LABEL[x]} − {LABEL[y]}" for x, y in PAIRS) + " |",
          "|---|" + "---|" * len(PAIRS)]
    for h in horizons:
        k = f"{h}w_auroc_avg"
        L.append(f"| {h}w | " + " | ".join(fmt(D[f'{x}-{y}'][k]['point'], D[f'{x}-{y}'][k]['ci'])
                                          for x, y in PAIRS) + " |")
    L += ["", "**Spread of AUROC (avg over tasks): mean ± SD over seeds (pooled folds) and over folds "
          "(seed-averaged)**", "", "| Horizon | H over seeds | B over seeds | H over folds | B over folds |",
          "|---|---|---|---|---|"]
    for i, h in enumerate(horizons):
        cell = lambda v, w: (f"{np.mean([r[i] for r in spread[v][w]]):.3f} ± "
                             f"{np.std([r[i] for r in spread[v][w]], ddof=1):.3f}")
        L.append(f"| {h}w | {cell('H', 'per_seed')} | {cell('B', 'per_seed')} | "
                 f"{cell('H', 'per_fold')} | {cell('B', 'per_fold')} |")
    L += ["", "Caveat (pre-registered): inputs were normalised with the original 149 training students, so "
          "some held-out students' inputs contributed to those means / SDs (inputs only, no labels; "
          "same for B and H).", ""]
    text = "\n".join(L)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "table.md"), "w", encoding="utf-8").write(text)
    json.dump({"n_boot": a.n_boot, "boot_seed": a.boot_seed, "decision": decision, "bootstrap": res,
               "spread": spread}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(text)
    print(f"saved {a.out}/table.md and results.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ces_forecast.npz")
    p.add_argument("--runs", default="runs/historyctl")
    p.add_argument("--out", default="runs/historyctl/eval")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--boot_seed", type=int, default=0)
    main(p.parse_args())
