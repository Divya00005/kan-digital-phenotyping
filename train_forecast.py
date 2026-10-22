"""
RQ1 step 3: train one forecasting model (KAN-ODE or the no-ODE control) on ces_forecast.npz.

    python train_forecast.py --model ode --version B --seed 0
    python train_forecast.py --run_all --seeds 0 1 2 3 4

Models (forecast_model.py):
    ode   = KANODEForecaster: encoder -> z(t0) -> dz/dt = KAN(z) -> heads at the target day
    noode = NoODEForecaster:  same encoder and KAN sizes, horizon given as input, no integration
  variant C (odernn_model.py, version B only, pre-registered in CLAUDE.md):
    odernn = ODERNNForecaster: ODE-RNN encoder (dh/dt = KAN(h) between days, GRU updates)
    grudt  = GRUDtForecaster:  GRU-delta-t control (no ODE), same no-ODE head
    python train_forecast.py --run_all --models grudt odernn --versions B --seeds 0
  history-only control (pre-registered in CLAUDE.md): version H = version B with the sensing
  inputs set to zero; 5-fold CV by student on the train + validation students (test not used):
    python train_forecast.py --run_all --models noode --versions B H --folds 0 1 2 3 4 \\
        --seeds 0 1 2 3 4 --out_root runs/historyctl
Input versions:
    A = sensing + engineered + static features
    B = A + symptom history known at t0 (today's scores, 28-day means, survey timing)

Only TRAINING and VALIDATION students are used here. The best epoch is chosen on the
validation set. Test predictions are made later, once, by evaluate_forecast.py --final.

Outputs in runs/forecast/<model>_<version>_seed<N>/:
    model.pt             weights of the best epoch
    history.json         per-epoch training loss and validation metrics
    results.json         settings, best epoch, validation metrics at the best epoch
    val_predictions.npz  validation predictions at the best epoch
"""
import argparse, json, os, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from forecast_model import KANODEForecaster, NoODEForecaster, forecast_loss
from odernn_model import ODERNNForecaster, GRUDtForecaster

TASKS = ["depression", "anxiety"]
CUTOFF = 3.0                    # screen-positive: subscore >= 3
SCORE_RANGE = (0.0, 6.0)        # PHQ-2 / GAD-2 subscores are 0..6
MODELS = {"ode": KANODEForecaster, "noode": NoODEForecaster,          # variant A
          "odernn": ODERNNForecaster, "grudt": GRUDtForecaster}        # variant C (version B only)


# ----------------------------------------------------------------------------- data
def load_data(path, device):
    """Load ces_forecast.npz. Model inputs go to `device`; labels stay as numpy."""
    d = np.load(path, allow_pickle=False)
    T = lambda k: torch.from_numpy(d[k]).to(device)
    keys = ["X", "E", "S", "H", "score", "mask", "offset_days", "current"]
    keys += [k for k in ["surv_scores", "surv_mask"] if k in d.files]       # variant C inputs
    data = {k: T(k) for k in keys}
    for k in ["score", "mask", "offset_days", "current", "uid", "date", "split", "horizons"]:
        data["np_" + k] = d[k]
    split, uid = d["split"], d["uid"]
    tr, va, te = (set(uid[split == s]) for s in (0, 1, 2))
    assert not (tr & va or tr & te or va & te), "student appears in two splits"
    return data


def build_model(name, version, data):
    X, E, S, H = data["X"], data["E"], data["S"], data["H"]
    h_in = H.shape[1] if version in ("B", "H") else 0
    return MODELS[name](X.shape[2], E.shape[1], S.shape[1], h_in)


def inputs(data, idx, version):
    """A = sensing only; B = sensing + history; H = history only: the version B model with the
    sensing inputs X and E replaced by zeros (history-only control, pre-registered 2026-10-22)."""
    x, e = data["X"][idx], data["E"][idx]
    if version == "H":
        x, e = torch.zeros_like(x), torch.zeros_like(e)
    h = data["H"][idx] if version in ("B", "H") else None
    return x, e, data["S"][idx], h, data["offset_days"][idx]


# ----------------------------------------------------------------------------- cross-validation
N_FOLDS, ES_FRACTION = 5, 0.1


def cv_folds(uid, split, n_folds=N_FOLDS):
    """History-only control (pre-registered): the train + validation students, sorted, shuffled with
    default_rng(0), cut into n_folds near-equal parts. Test students (split 2) are never included."""
    students = np.sort(np.unique(uid[split < 2]))
    return np.array_split(np.random.default_rng(0).permutation(students), n_folds)


def cv_rows(uid, split, fold, n_folds=N_FOLDS):
    """Row indices (train, early stopping, held out) for one fold. Early-stopping students = 10% of
    the other folds' students, drawn with default_rng(100 + fold)."""
    parts = cv_folds(uid, split, n_folds)
    rest = np.sort(np.concatenate([p for i, p in enumerate(parts) if i != fold]))
    es = np.random.default_rng(100 + fold).choice(rest, int(round(ES_FRACTION * len(rest))), replace=False)
    tv = split < 2
    held, stop = tv & np.isin(uid, parts[fold]), tv & np.isin(uid, es)
    return np.where(tv & ~held & ~stop)[0], np.where(stop)[0], np.where(held)[0]


def forward(model, data, idx, version, residual=False):
    """Model outputs (n, n_horizons, 4): [dep score, anx score, dep logit, anx logit].
    residual (version B only): the score outputs are a CHANGE added to today's score
    (known at t0), so the model starts from persistence and learns corrections."""
    if getattr(model, "uses_surveys", False):                            # variant C
        out = model(*inputs(data, idx, version), data["surv_scores"][idx], data["surv_mask"][idx])
    else:
        out = model(*inputs(data, idx, version))
    if residual:
        out = torch.cat([out[..., :2] + data["current"][idx].unsqueeze(1), out[..., 2:]], dim=-1)
    return out


@torch.no_grad()
def predict(model, data, rows, version, residual=False, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        idx = torch.as_tensor(rows[i:i + bs], device=data["X"].device)
        out.append(forward(model, data, idx, version, residual).float().cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- metrics
def val_metrics(pred, score, mask, horizons):
    """Validation summary used for monitoring: MAE (scores clipped to 0..6) and AUROC
    of the screen-positive logit, per horizon and task, plus their averages."""
    out, maes, aurocs = {}, [], []
    for k, h in enumerate(horizons):
        m = mask[:, k]
        for t, name in enumerate(TASKS):
            y = score[m, k, t]
            mae = float(np.abs(np.clip(pred[m, k, t], *SCORE_RANGE) - y).mean())
            lab = (y >= CUTOFF).astype(int)
            auroc = float(roc_auc_score(lab, pred[m, k, 2 + t])) if 0 < lab.sum() < len(lab) else float("nan")
            out[f"{h}w_{name}_mae"], out[f"{h}w_{name}_auroc"] = mae, auroc
            maes.append(mae); aurocs.append(auroc)
    out["mean_mae"], out["mean_auroc"] = float(np.mean(maes)), float(np.nanmean(aurocs))
    return out


def persistence_mae(current, score, mask):
    """Reference only: MAE of 'future score = today's score', averaged over horizons and tasks."""
    return float(np.mean([np.abs(current[mask[:, k], t] - score[mask[:, k], k, t]).mean()
                          for k in range(score.shape[1]) for t in range(2)]))


# ----------------------------------------------------------------------------- training
def train_one(a, model_name, version, seed, data, out_dir, fold=None):
    if a.residual and version != "B":
        raise ValueError("--residual needs today's score, which only input version B has")
    if getattr(MODELS[model_name], "uses_surveys", False) and version != "B":
        raise ValueError("variant C models are pre-registered for input version B only")
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    device = data["X"].device
    split = data["np_split"]
    if fold is None:                                   # fixed split: train / validation students
        tr, va, ho = np.where(split == 0)[0], np.where(split == 1)[0], None
    else:                                              # CV: train / early stopping / held-out fold
        tr, va, ho = cv_rows(data["np_uid"], split, fold)
        print(f"    fold {fold}: train {len(tr)} rows, early stopping {len(va)}, held out {len(ho)}")
    sc_va, mk_va = data["np_score"][va], data["np_mask"][va]
    horizons = data["np_horizons"].tolist()

    model = build_model(model_name, version, data).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    print(f"\n=== {model_name} | version {version} | seed {seed} | {n_params:,} parameters | "
          f"lr {a.lr} | residual {a.residual} | "
          f"train {len(tr)} val {len(va)} | device {device}")
    print(f"    reference: validation persistence MAE "
          f"{persistence_mae(data['np_current'][va], sc_va, mk_va):.3f}")

    va_t = {k: data[k][torch.as_tensor(va, device=device)] for k in ["score", "mask"]}
    g = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, wait, history = float("inf"), 0, None, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); total, n_seen = 0.0, 0
        perm = tr[torch.randperm(len(tr), generator=g).numpy()]
        batches = range(0, len(perm), a.batch)
        if a.max_batches:
            batches = list(batches)[:a.max_batches]
        for i in batches:
            idx = torch.as_tensor(perm[i:i + a.batch], device=device)
            loss = forecast_loss(forward(model, data, idx, version, a.residual),
                                 data["score"][idx], data["mask"][idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at epoch {epoch} (ODE blew up?)")
            opt.zero_grad(); loss.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            total += loss.item() * len(idx); n_seen += len(idx)

        pv = predict(model, data, va, version, a.residual)
        v_loss = forecast_loss(torch.from_numpy(pv).to(device), va_t["score"], va_t["mask"]).item()
        vm = val_metrics(pv, sc_va, mk_va, horizons)
        history.append({"epoch": epoch, "train_loss": total / n_seen, "val_loss": v_loss, **vm,
                        "seconds": time.time() - t0})
        crit = v_loss if a.select == "loss" else vm["mean_mae"]
        improved = crit < best
        print(f"epoch {epoch:3d} | train loss {total / n_seen:.4f} | val loss {v_loss:.4f} | "
              f"val MAE {vm['mean_mae']:.3f} | val AUROC {vm['mean_auroc']:.3f} | "
              f"{time.time() - t0:.0f}s{'  *' if improved else ''}", flush=True)
        if improved:
            best, best_epoch, wait = crit, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_pred = pv
        else:
            wait += 1
            if wait >= a.patience:
                print(f"early stop at epoch {epoch}: best val {a.select} {best:.4f} at epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    vm = val_metrics(best_pred, sc_va, mk_va, horizons)
    res = {"model": model_name, "version": version, "seed": seed, "fold": fold, "n_params": n_params,
           "best_epoch": best_epoch, "select_on": a.select, "best_val_criterion": best,
           "epochs_run": len(history), "val": vm,
           "val_persistence_mae": persistence_mae(data["np_current"][va], sc_va, mk_va),
           "settings": {k: v for k, v in vars(a).items()
                        if k not in ("run_all", "seeds", "versions", "models", "folds", "fold")},
           "torch": torch.__version__, "device": str(device)}
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    np.savez(os.path.join(out_dir, "val_predictions.npz"), pred=best_pred, score=sc_va, mask=mk_va,
             offset_days=data["np_offset_days"][va], current=data["np_current"][va],
             uid=data["np_uid"][va], date=data["np_date"][va])
    if ho is not None:                                 # out-of-fold predictions for the CV table
        np.savez(os.path.join(out_dir, "oof_predictions.npz"), rows=ho,
                 pred=predict(model, data, ho, version, a.residual), score=data["np_score"][ho],
                 mask=data["np_mask"][ho], current=data["np_current"][ho],
                 uid=data["np_uid"][ho], date=data["np_date"][ho])
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=2)   # written last = run complete
    print(f"best epoch {best_epoch}: val MAE {vm['mean_mae']:.3f} (persistence {res['val_persistence_mae']:.3f}), "
          f"val AUROC {vm['mean_auroc']:.3f} -> {out_dir}")
    return res


def main(a):
    device = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = False
    data = load_data(a.data, device)
    folds = a.folds if a.run_all else [a.fold]
    folds = folds if folds is not None else [None]
    if a.run_all:
        jobs = [(m, v, s, f) for f in folds for s in a.seeds for v in a.versions for m in a.models]
    else:
        jobs = [(a.model, a.version, a.seed, a.fold)]
    t_start = time.time()
    for n, (m, v, s, f) in enumerate(jobs, 1):
        name = f"{m}_{v}_seed{s}" if f is None else f"{m}_{v}_fold{f}_seed{s}"
        out_dir = os.path.join(a.out_root, name)
        if os.path.exists(os.path.join(out_dir, "results.json")) and not a.overwrite:
            print(f"[{n}/{len(jobs)}] skip {out_dir} (finished already; --overwrite to redo)")
            continue
        print(f"[{n}/{len(jobs)}] elapsed {(time.time() - t_start) / 60:.0f} min")
        train_one(a, m, v, s, data, out_dir, fold=f)
    print(f"\nall done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ces_forecast.npz")
    p.add_argument("--model", choices=list(MODELS), default="ode")
    p.add_argument("--version", choices=["A", "B", "H"], default="B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=None, help="CV fold 0-4 (history-only control)")
    p.add_argument("--run_all", action="store_true", help="every model x version for each of --seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--folds", type=int, nargs="+", default=None, help="for --run_all: CV folds")
    p.add_argument("--versions", nargs="+", choices=["A", "B", "H"], default=["A", "B"], help="for --run_all")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=["noode", "ode"],
                   help="for --run_all (variant C: --models grudt odernn --versions B)")
    p.add_argument("--residual", action="store_true",
                   help="version B: predict the change from today's score instead of the score")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping, 0 = off")
    p.add_argument("--select", choices=["loss", "mae"], default="loss",
                   help="validation criterion for the best epoch / early stopping")
    p.add_argument("--max_batches", type=int, default=0, help="debug: limit batches per epoch")
    p.add_argument("--device", default="auto")
    p.add_argument("--out_root", default="runs/forecast")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
