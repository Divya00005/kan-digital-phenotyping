"""
Train and evaluate KAN-MTLNet on the aligned CES dataset (paper Appendix B recipe).

    python train.py --data ces_kan_dataset.npz --seed 0 --out runs/seed0
"""
import argparse, json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from model import KANMTLNet, multitask_loss

TASKS = ["depression", "anxiety"]


def metrics(y, p):
    out = {}
    for t, name in enumerate(TASKS):
        out[f"{name}_auroc"] = roc_auc_score(y[:, t], p[:, t])
        out[f"{name}_auprc"] = average_precision_score(y[:, t], p[:, t])
        out[f"{name}_prevalence"] = float(y[:, t].mean())
    out["macro_auroc"] = np.mean([out[f"{n}_auroc"] for n in TASKS])
    out["macro_auprc"] = np.mean([out[f"{n}_auprc"] for n in TASKS])
    out["micro_auroc"] = roc_auc_score(y.ravel(), p.ravel())
    out["micro_auprc"] = average_precision_score(y.ravel(), p.ravel())
    return {k: float(v) for k, v in out.items()}


def student_bootstrap(y, p, uid, n_boot=1000, seed=0):
    """95% CI resampling STUDENTS (repeated surveys of one student are not independent)."""
    rng = np.random.default_rng(seed)
    students = np.unique(uid)
    rows = {s: np.where(uid == s)[0] for s in students}
    keys = ["depression_auroc", "anxiety_auroc", "macro_auroc",
            "depression_auprc", "anxiety_auprc", "macro_auprc"]
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rows[s] for s in rng.choice(students, len(students))])
        if y[idx].min(0).max() == 1 or y[idx].max(0).min() == 0:
            continue                                   # a task with only one class
        m = metrics(y[idx], p[idx])
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


@torch.no_grad()
def predict(model, X, E, S, bs=1024):
    model.eval()
    out = [torch.sigmoid(model(X[i:i + bs], E[i:i + bs], S[i:i + bs])) for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(os.cpu_count())
    d = np.load(a.data, allow_pickle=False)
    split, uid = d["split"], d["uid"]
    T = lambda arr: torch.from_numpy(arr)
    X, E, S, Y = T(d["X"]), T(d["E"]), T(d["S"]), T(d["Y"])
    tr, va, te = [np.where(split == k)[0] for k in (0, 1, 2)]
    assert not (set(uid[tr]) & set(uid[va]) or set(uid[tr]) & set(uid[te]) or set(uid[va]) & set(uid[te])), \
        "student appears in two splits"

    model = KANMTLNet(X.shape[2], E.shape[1], S.shape[1],
                      use_engineered=not a.no_engineered, use_static=not a.no_static)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"parameters: {n_params:,} | train {len(tr)} val {len(va)} test {len(te)}")

    best, best_state, wait, history = -1, None, 0, []
    g = torch.Generator().manual_seed(a.seed)
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); total = 0.0
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            loss = multitask_loss(model(X[b], E[b], S[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        pv = predict(model, X[va], E[va], S[va])
        mv = metrics(Y[va].numpy(), pv)
        history.append({"epoch": epoch, "train_loss": total / len(tr), **mv})
        print(f"epoch {epoch:3d} loss {total/len(tr):.4f} | val macro-AUROC {mv['macro_auroc']:.3f} "
              f"(dep {mv['depression_auroc']:.3f}, anx {mv['anxiety_auroc']:.3f}) | {time.time()-t0:.0f}s")
        if mv["macro_auroc"] > best:
            best, wait = mv["macro_auroc"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience:
                print(f"early stop at epoch {epoch} (best val macro-AUROC {best:.3f})")
                break

    model.load_state_dict(best_state)
    pt = predict(model, X[te], E[te], S[te])
    yt = Y[te].numpy()
    res = {"seed": a.seed, "n_params": n_params, "best_val_macro_auroc": best,
           "test": metrics(yt, pt), "test_ci95_student_bootstrap": student_bootstrap(yt, pt, uid[te]),
           "settings": vars(a)}
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    json.dump(history, open(os.path.join(a.out, "history.json"), "w"), indent=2)
    np.savez(os.path.join(a.out, "test_predictions.npz"), p=pt, y=yt, uid=uid[te], date=d["date"][te])
    m = res["test"]; ci = res["test_ci95_student_bootstrap"]
    print("\nTEST (held-out students)")
    for n in TASKS:
        print(f"  {n:10s} AUROC {m[n+'_auroc']:.3f} [{ci[n+'_auroc'][0]:.3f}, {ci[n+'_auroc'][1]:.3f}]  "
              f"AUC-PR {m[n+'_auprc']:.3f} (no-skill {m[n+'_prevalence']:.3f})")
    print(f"  macro      AUROC {m['macro_auroc']:.3f} [{ci['macro_auroc'][0]:.3f}, {ci['macro_auroc'][1]:.3f}]  "
          f"AUC-PR {m['macro_auprc']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ces_kan_dataset.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--no_static", action="store_true")
    p.add_argument("--no_engineered", action="store_true")
    p.add_argument("--out", default="runs/seed0")
    main(p.parse_args())
