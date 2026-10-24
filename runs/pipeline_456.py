"""
RQ1 steps 4-6, run unattended (approved by the student on 2026-10-22). All rules fixed in advance:

Step 4 (validation only), settings tried with seed 0:
    C0  current settings (lr 1e-3)            - already trained in step 3 (runs/forecast)
    C1  lr 3e-4                               - versions A and B
    C2  residual (change from today), lr 1e-3 - version B only (A has no today's score)
    C3  residual, lr 3e-4                     - version B only
  Choice per version: lowest validation MAE (mean over the 4 horizons and both tasks),
  averaged over the KAN-ODE and no-ODE models, so both models always share one setting.
  Settings within 0.005 of the best count as tied -> keep the simpler one (order C0, C1, C2, C3).
Step 5: chosen setting, seeds 0-4, both models -> runs/final (seed 0 copied from step 4).
        Then the validation table for the 5-seed ensembles.
Step 6: evaluate_forecast.py --split test --final, once.
Any failure stops the pipeline before the test set is touched.

    python runs/pipeline_456.py            # real run
    python runs/pipeline_456.py --smoke    # tiny runs in D:\\tmp, never touches the test set
"""
import json, os, shutil, subprocess, sys, time
import numpy as np

SMOKE = "--smoke" in sys.argv
PY = sys.executable
ENV = dict(os.environ, TEMP=r"D:\tmp", TMP=r"D:\tmp", TMPDIR=r"D:\tmp", PYTHONUNBUFFERED="1")
BASE = r"D:\tmp\pipe_smoke" if SMOKE else "runs"
CONFIGS = {  # name: (extra training flags, run folder, versions)
    "C0": ([], r"D:\tmp\smoke" if SMOKE else "runs/forecast", ["A", "B"]),
    "C1": (["--lr", "3e-4"], f"{BASE}/tune/C1", ["A", "B"]),
    "C2": (["--residual", "--lr", "1e-3"], f"{BASE}/tune/C2", ["B"]),
    "C3": (["--residual", "--lr", "3e-4"], f"{BASE}/tune/C3", ["B"]),
}
ORDER, TIE = ["C0", "C1", "C2", "C3"], 0.005
SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
FINAL = f"{BASE}/final"
EXTRA = ["--epochs", "1", "--max_batches", "2"] if SMOKE else []


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M')}  {msg}", flush=True)


def run(cmd, logfile):
    log(" ".join(cmd) + f"   (log: {logfile})")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    if r.returncode:
        log(f"FAILED (exit {r.returncode}); see {logfile}. Pipeline stopped; test set NOT touched.")
        sys.exit(r.returncode)


def val_mae(root, v):
    vals = {m: json.load(open(f"{root}/{m}_{v}_seed0/results.json"))["val"]["mean_mae"] for m in ("noode", "ode")}
    return float(np.mean(list(vals.values()))), vals


t0 = time.time()
log(f"START steps 4-6 {'(SMOKE TEST)' if SMOKE else ''}")

# ------------------------------------------------------------------ step 4
log("STEP 4: tuning on validation students (seed 0)")
for name in ["C1", "C2", "C3"]:
    flags, root, versions = CONFIGS[name]
    run([PY, "train_forecast.py", "--run_all", "--seeds", "0", "--versions", *versions,
         *flags, *EXTRA, "--out_root", root], f"{root}.log")

selection = {}
for v in ["A", "B"]:
    scores = {c: val_mae(CONFIGS[c][1], v) for c in ORDER if v in CONFIGS[c][2]}
    best = min(s for s, _ in scores.values())
    chosen = next(c for c in ORDER if c in scores and scores[c][0] <= best + TIE)
    selection[v] = {"chosen": chosen, "flags": CONFIGS[chosen][0], "best_val_mae": best,
                    "scores": {c: {"mean_val_mae": s, "per_model": per} for c, (s, per) in scores.items()}}
    log(f"version {v}: " + ", ".join(f"{c} {s:.4f}" for c, (s, _) in scores.items()) + f" -> chosen {chosen}")
os.makedirs(f"{BASE}/tune", exist_ok=True)
json.dump({"rule": "lowest mean val MAE over both models; ties within 0.005 -> simpler (C0<C1<C2<C3)",
           "selection": selection}, open(f"{BASE}/tune/selection.json", "w"), indent=2)

# ------------------------------------------------------------------ step 5
log(f"STEP 5: final training, seeds {SEEDS}")
for v in ["A", "B"]:
    flags, root, _ = CONFIGS[selection[v]["chosen"]]
    for m in ("noode", "ode"):
        src, dst = f"{root}/{m}_{v}_seed0", f"{FINAL}/{m}_{v}_seed0"
        if not os.path.exists(f"{dst}/results.json"):
            shutil.copytree(src, dst, dirs_exist_ok=True)
    run([PY, "train_forecast.py", "--run_all", "--seeds", *map(str, SEEDS), "--versions", v,
         *flags, *EXTRA, "--out_root", FINAL], f"{FINAL}/train_{v}.log")
run([PY, "evaluate_forecast.py", "--runs", FINAL, "--seeds", *map(str, SEEDS),
     "--out", f"{FINAL}/eval_val"], f"{FINAL}/eval_val.log")

# ------------------------------------------------------------------ step 6
if SMOKE:
    log("STEP 6 skipped in smoke mode (test set not touched)")
else:
    log("STEP 6: test set, evaluated once")
    run([PY, "evaluate_forecast.py", "--runs", FINAL, "--split", "test", "--final",
         "--seeds", *map(str, SEEDS), "--out", f"{FINAL}/eval_test"], f"{FINAL}/eval_test.log")
log(f"DONE in {(time.time() - t0) / 3600:.1f} h")
