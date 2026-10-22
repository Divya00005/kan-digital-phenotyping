"""Tests for the RQ3 training code (pre-registered in CLAUDE.md). Run: python test_rq3_training.py
Uses only student IDs / splits from data/ces_forecast.npz (fold tests); models use synthetic inputs."""
import numpy as np
import torch
from train_forecast import build_model, forward, cv_folds, rq3_students, rq3_rows, RQ3_ROLES
from quantile_model import QuantileNoODEForecaster, pinball_loss, sort_quantiles, quantile_index, QUANTILES
from baseline_history import fit_predict_quantile

torch.manual_seed(0)
D = np.load("data/ces_forecast.npz")
UID, SPLIT = D["uid"], D["split"]
TEST_STUDENTS = set(UID[SPLIT == 2])


def test_nested_folds():
    """Per outer fold: 34 evaluation students never in that round's training, early-stopping or
    calibration sets; split = 68 model (61 train + 7 stop) + 68 calibration; inner folds = 4 x 34
    calibration folds covering the 136 other students, each with 92 train + 10 stop; no test students;
    reproducible."""
    parts = cv_folds(UID, SPLIT)
    for k in range(5):
        rest = set(np.concatenate([p for i, p in enumerate(parts) if i != k]))
        inner_cals = []
        for role in RQ3_ROLES:
            s = rq3_students(UID, SPLIT, k, role)
            again = rq3_students(UID, SPLIT, k, role)
            assert all(np.array_equal(s[x], again[x]) for x in s), "not reproducible"
            ev, tr, st, cal, model = (set(s[x]) for x in ["eval", "train", "stop", "cal", "model"])
            assert ev == set(parts[k]) and len(ev) == 34
            assert not ev & (tr | st | cal), f"outer {k} {role}: an evaluation student is used"
            assert not tr & st and not (tr | st) & cal and tr | st == model
            assert model | cal == rest and not (model | cal | ev) & TEST_STUDENTS
            if role == "split":
                assert (len(tr), len(st), len(cal)) == (61, 7, 68), (len(tr), len(st), len(cal))
            else:
                assert (len(tr), len(st), len(cal)) == (92, 10, 34), (len(tr), len(st), len(cal))
                inner_cals.append(cal)
        assert set().union(*inner_cals) == rest and sum(len(c) for c in inner_cals) == 136
    print("ok  nested folds: 34 evaluation students per round, never in training / early stopping / "
          "calibration; split 61 + 7 / 68; inner folds 4 x 34 (92 + 10 train); no test students; reproducible")


def test_rows_match_students():
    for k in range(5):
        for role in ["split", "inner2"]:
            s, r = rq3_students(UID, SPLIT, k, role), rq3_rows(UID, SPLIT, k, role)
            for part in s:
                assert set(UID[r[part]]) == set(s[part]) and (SPLIT[r[part]] < 2).all()
    print("ok  row indices match the student sets and contain no test-set rows")


def fake_data(n=6):
    keep = torch.ones(10); keep[:9] = 0                      # 9 gender / race columns + ios_share
    return {"X": torch.randn(n, 60, 22), "E": torch.randn(n, 34), "S": torch.randn(n, 10),
            "H": torch.randn(n, 6), "offset_days": torch.tensor([[7., 14., 28., 56.]]).repeat(n, 1),
            "current": torch.randint(0, 7, (n, 2)).float(), "S_keep": keep}


def test_Hn_ignores_demographics():
    """Hn's output does not change when gender / race inputs (or sensing) change, but does when
    ios_share changes; H's output does change with gender / race."""
    d1 = fake_data()
    d_demo = dict(d1, S=d1["S"].clone()); d_demo["S"][:, :9] = torch.randn(6, 9) * 5
    d_sens = dict(d1, X=torch.randn_like(d1["X"]) * 5, E=torch.randn_like(d1["E"]) * 5)
    d_ios = dict(d1, S=d1["S"].clone()); d_ios["S"][:, 9] += 3
    idx = torch.arange(6)
    diff = lambda m, v, a, b: (forward(m, a, idx, v) - forward(m, b, idx, v)).abs().max().item()
    with torch.no_grad():
        mHn, mH = build_model("noode", "Hn", d1).eval(), build_model("noode", "H", d1).eval()
        assert diff(mHn, "Hn", d1, d_demo) == 0 and diff(mHn, "Hn", d1, d_sens) == 0
        assert diff(mHn, "Hn", d1, d_ios) > 1e-4 and diff(mH, "H", d1, d_demo) > 1e-4
    print("ok  version Hn ignores gender / race (and sensing) but uses ios_share; version H uses gender / race")


def test_pinball_loss():
    """Matches the formula; a constant minimising it is the empirical quantile."""
    rng = np.random.default_rng(0)
    y = torch.tensor(rng.normal(size=(50, 4, 2)), dtype=torch.float32)
    p = torch.tensor(rng.normal(size=(50, 4, 8)), dtype=torch.float32)
    mask = torch.tensor(rng.random((50, 4)) < 0.8)
    q = np.array(QUANTILES)
    u = y.numpy()[mask.numpy()][:, :, None] - p.numpy()[mask.numpy()].reshape(-1, 2, 4)
    manual = np.maximum(q * u, (q - 1) * u).mean()
    assert np.isclose(pinball_loss(p, y, mask).item(), manual, atol=1e-6)
    sample = rng.exponential(size=4000)
    grid = np.linspace(0, 5, 2001)
    for level in QUANTILES:
        losses = [np.mean(np.maximum(level * (sample - c), (level - 1) * (sample - c))) for c in grid]
        assert abs(grid[int(np.argmin(losses))] - np.quantile(sample, level)) < 0.02, level
    print("ok  pinball loss matches the formula; its minimiser is the empirical quantile")


def test_quantile_model_and_sorting():
    d = fake_data(8)
    m = QuantileNoODEForecaster(22, 34, 10, 6)
    out = m(d["X"], d["E"], d["S"], d["H"], d["offset_days"])
    assert out.shape == (8, 4, 8)
    s = sort_quantiles(np.random.default_rng(1).normal(size=(5, 4, 8)))
    assert np.all(np.diff(s.reshape(5, 4, 2, 4), axis=-1) >= 0)
    assert quantile_index(0, 0.05) == 0 and quantile_index(1, 0.95) == 7 and quantile_index(1, 0.10) == 5
    print("ok  quantile model outputs (batch, 4 horizons, 8); sorting removes crossings; column layout")


def test_quantile_model_learns():
    """On a small synthetic batch, trained quantiles bracket the data at roughly the right rates."""
    torch.manual_seed(0)
    d = fake_data(64)
    y = (2 + d["H"][:, :1, None] + 0.5 * torch.randn(64, 4, 2)).clamp(0, 6)
    mask = torch.ones(64, 4, dtype=torch.bool)
    m = QuantileNoODEForecaster(22, 34, 10, 6)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        loss = pinball_loss(m(d["X"], d["E"], d["S"], d["H"], d["offset_days"]), y, mask)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = m(d["X"], d["E"], d["S"], d["H"], d["offset_days"]).reshape(64, 4, 2, 4)
    below = (y.unsqueeze(-1) <= p).float().mean((0, 1, 2)).numpy()       # share of y below each quantile
    assert below[0] < 0.2 and below[3] > 0.8 and np.all(np.diff(below) > 0), below
    print(f"ok  quantile KAN learns: share of y below the 5/10/90/95% outputs = {[round(float(b), 2) for b in below]}")


def test_linear_quantile_regression():
    """Linear quantile regression on heteroscedastic synthetic data: the 5-95% band covers about 90%
    of new data; outputs are sorted."""
    rng = np.random.default_rng(2)
    n = 1500
    F = rng.normal(size=(n, 3))
    sd = 0.5 + np.abs(F[:, 0])
    y = np.clip(2 + F[:, 1] + sd * rng.normal(size=n), -10, 10)
    score = np.repeat(np.repeat(y[:, None, None], 4, 1), 2, 2)
    mask = np.ones((n, 4), bool)
    F_new = rng.normal(size=(1000, 3))
    y_new = 2 + F_new[:, 1] + (0.5 + np.abs(F_new[:, 0])) * rng.normal(size=1000)
    q = fit_predict_quantile(F, score, mask, F_new)
    cov = np.mean((y_new >= q[:, 0, 0]) & (y_new <= q[:, 0, 3]))
    assert 0.83 <= cov <= 0.95 and np.all(np.diff(q.reshape(1000, 4, 2, 4), axis=-1) >= 0), cov
    print(f"ok  linear quantile regression: 5-95% band covers {cov:.2f} of new synthetic data; sorted")


if __name__ == "__main__":
    test_nested_folds()
    test_rows_match_students()
    test_Hn_ignores_demographics()
    test_pinball_loss()
    test_quantile_model_and_sorting()
    test_quantile_model_learns()
    test_linear_quantile_regression()
    print("all tests passed")
