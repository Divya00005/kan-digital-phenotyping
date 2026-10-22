"""
RQ1 step 3: train one forecasting model (KAN-ODE or the no-ODE control) on ces_forecast.npz.

    python train_forecast.py --model ode --version B --seed 0
    python train_forecast.py --run_all --seeds 0 1 2 3 4

Models (forecast_model.py):
    ode   = KANODEForecaster: encoder -> z(t0) -> dz/dt = KAN(z) -> heads at the target day
    noode = NoODEForecaster:  same encoder and KAN sizes, horizon given as input, no integration
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

TASKS = ["depression", "anxiety"]
CUTOFF = 3.0                    # screen-positive: subscore >= 3
SCORE_RANGE = (0.0, 6.0)        # PHQ-2 / GAD-2 subscores are 0..6
MODELS = {"ode": KANODEForecaster, "noode": NoODEForecaster}


# ----------------------------------------------------------------------------- data
def load_data(path, device):
    """Load ces_forecast.npz. Model inputs go to `device`; labels stay as numpy."""
    d = np.load(path, allow_pickle=False)
    T = lambda k: torch.from_numpy(d[k]).to(device)
    data = {k: T(k) for k in ["X", "E", "S", "H", "score", "mask", "offset_days", "current"]}
    for k in ["score", "mask", "offset_days", "current", "uid", "date", "split", "horizons"]:
        data["np_" + k] = d[k]
    split, uid = d["split"], d["uid"]
    tr, va, te = (set(uid[split == s]) for s in (0, 1, 2))
    assert not (tr & va or tr & te or va & te), "student appears in two splits"
    return data


def build_model(name, version, data):
    X, E, S, H = data["X"], data["E"], data["S"], data["H"]
    h_in = H.shape[1] if version == "B" else 0
    return MODELS[name](X.shape[2], E.shape[1], S.shape[1], h_in)


def inputs(data, idx, version):
    h = data["H"][idx] if version == "B" else None
    return data["X"][idx], data["E"][idx], data["S"][idx], h, data["offset_days"][idx]


def forward(model, data, idx, version, residual=False):
    """Model outputs (n, n_horizons, 4): [dep score, anx score, dep logit, anx logit].
    residual (version B only): the score outputs are a CHANGE added to today's score
    (known at t0), so the model starts from persistence and learns corrections."""
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
def train_one(a, model_name, version, seed, data, out_dir):
    if a.residual and version != "B":
        raise ValueError("--residual needs today's score, which only input version B has")
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    device = data["X"].device
    split = data["np_split"]
    tr, va = np.where(split == 0)[0], np.where(split == 1)[0]
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
    res = {"model": model_name, "version": version, "seed": seed, "n_params": n_params,
           "best_epoch": best_epoch, "select_on": a.select, "best_val_criterion": best,
           "epochs_run": len(history), "val": vm,
           "val_persistence_mae": persistence_mae(data["np_current"][va], sc_va, mk_va),
           "settings": {k: v for k, v in vars(a).items() if k not in ("run_all", "seeds", "versions")},
           "torch": torch.__version__, "device": str(device)}
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
    np.savez(os.path.join(out_dir, "val_predictions.npz"), pred=best_pred, score=sc_va, mask=mk_va,
             offset_days=data["np_offset_days"][va], current=data["np_current"][va],
             uid=data["np_uid"][va], date=data["np_date"][va])
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=2)   # written last = run complete
    print(f"best epoch {best_epoch}: val MAE {vm['mean_mae']:.3f} (persistence {res['val_persistence_mae']:.3f}), "
          f"val AUROC {vm['mean_auroc']:.3f} -> {out_dir}")
    return res


def main(a):
    device = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = False
    data = load_data(a.data, device)
    if a.run_all:
        jobs = [(m, v, s) for s in a.seeds for v in a.versions for m in ("noode", "ode")]
    else:
        jobs = [(a.model, a.version, a.seed)]
    t_start = time.time()
    for n, (m, v, s) in enumerate(jobs, 1):
        out_dir = os.path.join(a.out_root, f"{m}_{v}_seed{s}")
        if os.path.exists(os.path.join(out_dir, "results.json")) and not a.overwrite:
            print(f"[{n}/{len(jobs)}] skip {out_dir} (finished already; --overwrite to redo)")
            continue
        print(f"[{n}/{len(jobs)}] elapsed {(time.time() - t_start) / 60:.0f} min")
        train_one(a, m, v, s, data, out_dir)
    print(f"\nall done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ces_forecast.npz")
    p.add_argument("--model", choices=list(MODELS), default="ode")
    p.add_argument("--version", choices=["A", "B"], default="B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_all", action="store_true", help="every model x version for each of --seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--versions", nargs="+", choices=["A", "B"], default=["A", "B"], help="for --run_all")
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
