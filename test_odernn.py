"""Tests for the RQ1 variant C sequence models (odernn_model.py). Run: python test_odernn.py"""
import torch
from ode import rk4_trajectory
from forecast_model import forecast_loss, DAY
from odernn_model import ODERNNForecaster, GRUDtForecaster, N_DAYS

torch.manual_seed(0)
MODELS = (ODERNNForecaster, GRUDtForecaster)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fake_batch(B=8, p_valid=0.85, p_survey=0.2):
    x = torch.randn(B, N_DAYS - 1, 22)
    x[..., -1] = (torch.rand(B, N_DAYS - 1) < p_valid).float()       # valid-day flag
    e, s, h = torch.randn(B, 34), torch.randn(B, 10), torch.randn(B, 6)
    sm = torch.rand(B, N_DAYS) < p_survey
    sm[:, -1] = True                                                  # every cutoff is a survey day
    sv = torch.randint(0, 7, (B, N_DAYS, 2)).float() / 6 * sm.unsqueeze(-1)
    offs = torch.tensor([7, 14, 28, 56]).repeat(B, 1).float() + torch.randint(-3, 4, (B, 4))
    score = torch.randint(0, 7, (B, 4, 2)).float()
    return x, e, s, h, offs, sv, sm, score


def test_parameter_counts_matched():
    n = {cls.__name__: sum(p.numel() for p in cls(22, 34, 10, 6).parameters()) for cls in MODELS}
    a, b = n["ODERNNForecaster"], n["GRUDtForecaster"]
    assert abs(a - b) / a < 0.01, n
    print(f"ok  parameter counts match: ODE-RNN {a:,} vs GRU-dt {b:,} ({100 * (b - a) / a:+.2f}%)")


def test_missing_days_are_ignored():
    """Values on invalid sensing days and on days without a survey must not change the output."""
    for cls in MODELS:
        m = cls(22, 34, 10, 6).eval()
        x, e, s, h, offs, sv, sm, _ = fake_batch()
        x2, sv2 = x.clone(), sv.clone()
        inv = x[..., -1] == 0
        x2[..., :-1][inv] = torch.randn(int(inv.sum()), 21) * 10          # garbage on invalid days
        sv2[~sm] = torch.rand(int((~sm).sum()), 2)                         # garbage where no survey
        with torch.no_grad():
            d = (m(x, e, s, h, offs, sv, sm) - m(x2, e, s, h, offs, sv2, sm)).abs().max().item()
        assert d == 0.0, (cls.__name__, d)
    print("ok  invalid sensing days and days without a survey are ignored by both models")


def test_empty_walk_is_pure_ode():
    """With no observations at all, the ODE-RNN state is just the ODE solution from 0 over 60 days."""
    m = ODERNNForecaster(22, 34, 10, 6).eval()
    x, _, _, _, _, sv, sm, _ = fake_batch(4)
    x[..., -1] = 0
    sm[:] = False
    with torch.no_grad():
        ours = m.walk(x, sv, sm)
        ref = rk4_trajectory(m.f, torch.zeros(4, m.hidden), N_DAYS - 1, DAY)[:, -1]
    err = (ours - ref).abs().max().item()
    assert err < 1e-6 and ref.abs().max() > 1e-3, err
    print(f"ok  no observations -> ODE-RNN state = 60-day RK4 solution (error {err:.1e})")


def test_zero_dynamics_is_plain_gru():
    """If dh/dt = 0, the ODE-RNN must reduce to plain GRU updates on the observed days."""
    m = ODERNNForecaster(22, 34, 10, 6).eval()
    with torch.no_grad():
        for p in m.f.parameters():
            p.zero_()
    x, _, _, _, _, sv, sm, _ = fake_batch(4)
    with torch.no_grad():
        ours = m.walk(x, sv, sm)
        ref = torch.zeros(4, m.hidden)
        for b in range(4):
            hb = torch.zeros(1, m.hidden)
            for k in range(N_DAYS):
                if k < N_DAYS - 1 and x[b, k, -1] > 0.5:
                    hb = m.gru_x(x[b:b + 1, k], hb)
                if sm[b, k]:
                    hb = m.gru_s(sv[b:b + 1, k], hb)
            ref[b] = hb[0]
    err = (ours - ref).abs().max().item()
    assert err < 1e-6, err
    print(f"ok  zero dynamics -> ODE-RNN = plain GRU over the observed days (error {err:.1e})")


def test_control_time_gaps():
    """GRU-dt must receive the exact time since the previous update (in weeks)."""
    m = GRUDtForecaster(22, 34, 10, 6).eval()
    x, _, _, _, _, sv, sm, _ = fake_batch(1)
    x[..., -1] = 0                                     # no sensing updates
    sm[:] = False
    sm[0, [10, 25, 60]] = True                         # surveys on days 10, 25 and t0
    seen = []
    hook = m.gru_s.register_forward_hook(lambda mod, inp, out: seen.append(inp[0][0, -1].item()))
    with torch.no_grad():
        m.walk(x, sv, sm)
    hook.remove()
    # gru_s runs every day (masked by torch.where); keep the calls on the survey days
    got = [seen[k] for k in (10, 25, 60)]
    want = [10 * DAY, 15 * DAY, 35 * DAY]
    assert all(abs(g - w) < 1e-6 for g, w in zip(got, want)), (got, want)
    print(f"ok  GRU-dt gets the gaps since the previous update: {[round(g * 7) for g in got]} days")


def test_timing_matters():
    """The same single survey on day 30 vs day 50 must give different states (both models)."""
    for cls in MODELS:
        m = cls(22, 34, 10, 6).eval()
        x, _, _, _, _, sv, sm, _ = fake_batch(1)
        x[..., -1] = 0
        sm[:] = False
        sv[:] = 0.5
        a_m, b_m = sm.clone(), sm.clone()
        a_m[0, 30] = True; b_m[0, 50] = True
        with torch.no_grad():
            d = (m.walk(x, sv, a_m) - m.walk(x, sv, b_m)).abs().max().item()
        assert d > 1e-4, (cls.__name__, d)
    print("ok  both models are sensitive to WHEN a survey happened")


def test_batch_independence():
    """Each sample's output must not depend on the other samples in the batch."""
    for cls in MODELS:
        m = cls(22, 34, 10, 6).eval()
        x, e, s, h, offs, sv, sm, _ = fake_batch(6)
        with torch.no_grad():
            full = m(x, e, s, h, offs, sv, sm)
            one = torch.cat([m(x[i:i + 1], e[i:i + 1], s[i:i + 1], h[i:i + 1], offs[i:i + 1],
                               sv[i:i + 1], sm[i:i + 1]) for i in range(6)])
        assert torch.allclose(full, one, atol=1e-5), cls.__name__
    print("ok  outputs are the same per sample alone or in a batch")


def test_gradients_reach_all_parts():
    for cls in MODELS:
        m = cls(22, 34, 10, 6)
        x, e, s, h, offs, sv, sm, score = fake_batch(8)
        forecast_loss(m(x, e, s, h, offs, sv, sm), score, torch.ones(8, 4, dtype=torch.bool)).backward()
        g = {name: sum(p.grad.abs().sum() for p in getattr(m, name).parameters()).item()
             for name in ["f", "gru_x", "gru_s", "shared", "static", "g", "head"]}
        assert all(v > 0 for v in g.values()), (cls.__name__, g)
    print("ok  gradients reach the dynamics KAN, both GRU cells, the mixing KANs and the head")


def test_models_can_fit_small_batch():
    """Both models must be able to memorise 32 samples x 4 horizons."""
    for cls in MODELS:
        torch.manual_seed(0)
        m = cls(22, 34, 10, 6).to(DEV)
        batch = [t.to(DEV) for t in fake_batch(32)]
        x, e, s, h, offs, sv, sm, score = batch
        mask = torch.ones(32, 4, dtype=torch.bool, device=DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        for _ in range(400):
            loss = forecast_loss(m(x, e, s, h, offs, sv, sm), score, mask, lam=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            mae = (m(x, e, s, h, offs, sv, sm)[..., :2] - score).abs().mean().item()
        assert mae < 0.5, (cls.__name__, mae)
        print(f"ok  {cls.__name__} memorises a small batch (MAE {mae:.2f}, device {DEV})")


if __name__ == "__main__":
    test_parameter_counts_matched()
    test_missing_days_are_ignored()
    test_empty_walk_is_pure_ode()
    test_zero_dynamics_is_plain_gru()
    test_control_time_gaps()
    test_timing_matters()
    test_batch_independence()
    test_gradients_reach_all_parts()
    test_models_can_fit_small_batch()
    print("all tests passed")
