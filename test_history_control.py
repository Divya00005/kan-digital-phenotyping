"""Tests for the history-only control (pre-registered in CLAUDE.md). Run: python test_history_control.py
Uses the student IDs / splits of data/ces_forecast.npz for the fold tests."""
import numpy as np
import torch
from train_forecast import build_model, forward, cv_folds, cv_rows, N_FOLDS
from baseline_history import fit_predict

torch.manual_seed(0)
D = np.load("data/ces_forecast.npz")
UID, SPLIT = D["uid"], D["split"]


def test_folds_complete_and_disjoint():
    parts = cv_folds(UID, SPLIT)
    allp = np.concatenate(parts)
    tv_students = set(UID[SPLIT < 2])
    assert len(parts) == N_FOLDS and [len(p) for p in parts] == [34] * 5, [len(p) for p in parts]
    assert len(allp) == len(set(allp)) == len(tv_students) == 170 and set(allp) == tv_students
    assert not set(allp) & set(UID[SPLIT == 2]), "a test student is in a fold"
    assert all(np.array_equal(a, b) for a, b in zip(parts, cv_folds(UID, SPLIT))), "folds not reproducible"
    print("ok  5 folds x 34 students = all 170 train+val students, no overlap, no test students, reproducible")


def test_fold_rows():
    tv_rows = set(np.where(SPLIT < 2)[0])
    held_all = []
    for f in range(N_FOLDS):
        tr, es, ho = cv_rows(UID, SPLIT, f)
        s_tr, s_es, s_ho = set(UID[tr]), set(UID[es]), set(UID[ho])
        assert not (s_tr & s_es or s_tr & s_ho or s_es & s_ho), f"fold {f}: a student in two roles"
        assert len(s_es) == 14 and len(s_ho) == 34 and len(s_tr) == 122, (len(s_tr), len(s_es), len(s_ho))
        assert set(tr) | set(es) | set(ho) == tv_rows, f"fold {f}: rows missing"
        assert (SPLIT[np.concatenate([tr, es, ho])] < 2).all(), f"fold {f}: a test row is used"
        held_all.append(ho)
    held_all = np.concatenate(held_all)
    assert len(held_all) == len(set(held_all)) == len(tv_rows), "held-out rows must cover train+val once"
    print("ok  each fold: 122 train / 14 early-stopping / 34 held-out students, disjoint; "
          "held-out folds cover every train+val row exactly once")


def fake_data(n=6):
    return {"X": torch.randn(n, 60, 22), "E": torch.randn(n, 34), "S": torch.randn(n, 10),
            "H": torch.randn(n, 6), "offset_days": torch.tensor([[7., 14., 28., 56.]]).repeat(n, 1),
            "current": torch.randint(0, 7, (n, 2)).float()}


def test_version_H_ignores_sensing():
    """Changing the sensing inputs X and E must not change a version H output (it does for B)."""
    d1 = fake_data()
    d2 = dict(d1, X=torch.randn_like(d1["X"]) * 5, E=torch.randn_like(d1["E"]) * 5)
    idx = torch.arange(6)
    for version, should_change in [("H", False), ("B", True)]:
        m = build_model("noode", version, d1).eval()
        with torch.no_grad():
            diff = (forward(m, d1, idx, version) - forward(m, d2, idx, version)).abs().max().item()
        assert (diff > 1e-4) == should_change, (version, diff)
    print("ok  version H output is unchanged when sensing changes (version B output does change)")


def test_same_model_size_B_and_H():
    d = fake_data()
    nB = sum(p.numel() for p in build_model("noode", "B", d).parameters())
    nH = sum(p.numel() for p in build_model("noode", "H", d).parameters())
    assert nB == nH, (nB, nH)
    print(f"ok  versions B and H have the same architecture and size ({nB:,} parameters)")


def test_baseline_fit_predict():
    """Shapes, finite values, and the logistic part learns an obvious signal."""
    rng = np.random.default_rng(0)
    F_tr, F_ho = rng.normal(size=(400, 16)), rng.normal(size=(100, 16))
    score_tr = np.clip(np.round(3 + 2 * F_tr[:, :1, None] + rng.normal(size=(400, 4, 2)) * 0.5), 0, 6)
    mask_tr = rng.random((400, 4)) < 0.9
    p = fit_predict(F_tr, score_tr, mask_tr, F_ho)
    assert p.shape == (100, 4, 4) and np.isfinite(p).all()
    corr = np.corrcoef(F_ho[:, 0], p[:, 0, 2])[0, 1]
    assert corr > 0.9, corr
    print(f"ok  simple baseline: (n, 4 horizons, 4 outputs), finite, logit follows the signal (r = {corr:.2f})")


if __name__ == "__main__":
    test_folds_complete_and_disjoint()
    test_fold_rows()
    test_version_H_ignores_sensing()
    test_same_model_size_B_and_H()
    test_baseline_fit_predict()
    print("all tests passed")
