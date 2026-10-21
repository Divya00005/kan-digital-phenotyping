"""Tests for the RK4 solver and the RQ1 forecasting models. Run: python test_ode.py"""
import math
import torch
from ode import rk4_trajectory
from forecast_model import (KANODEForecaster, NoODEForecaster, KANDynamics,
                            forecast_loss, DAY, MAX_DAYS)

torch.manual_seed(0)


def test_rk4_exact_solution():
    """dz/dt = -z has the exact solution z(t) = z0 * exp(-t)."""
    z0 = torch.tensor([[1.0, -2.0]])
    traj = rk4_trajectory(lambda z: -z, z0, n_steps=56, dt=DAY)       # 8 weeks
    exact = z0 * math.exp(-56 * DAY)
    err = (traj[:, -1] - exact).abs().max().item()
    assert err < 1e-5, err
    print(f"ok  RK4 matches exp(-t) after 8 weeks (error {err:.1e})")


def test_rk4_matches_torchdiffeq():
    """Our solver vs an adaptive Dormand-Prince solver on a random KAN dynamics."""
    from torchdiffeq import odeint
    f = KANDynamics(latent=16, width=32)
    z0 = torch.randn(4, 16)
    with torch.no_grad():
        ours = rk4_trajectory(f, z0, MAX_DAYS, DAY)[:, -1]
        ref = odeint(lambda t, z: f(z), z0, torch.tensor([0.0, MAX_DAYS * DAY]),
                     method="dopri5", rtol=1e-7, atol=1e-8)[-1]
    err = (ours - ref).abs().max().item()
    assert err < 1e-3, err
    print(f"ok  RK4 agrees with torchdiffeq dopri5 (max difference {err:.1e})")


def fake_batch(B=32, h_in=6):
    x, e, s = torch.randn(B, 60, 22), torch.randn(B, 34), torch.randn(B, 10)
    h = torch.randn(B, h_in)
    offs = torch.tensor([7, 14, 28, 56]).repeat(B, 1).float() + torch.randint(-3, 4, (B, 4))
    score = torch.randint(0, 7, (B, 4, 2)).float()
    mask = torch.ones(B, 4, dtype=torch.bool)
    return x, e, s, h, offs, score, mask


def test_offset_zero_is_initial_state():
    """With offset 0 the ODE model must read exactly z(t0), so time indexing is correct."""
    m = KANODEForecaster(22, 34, 10, h_in=6)
    x, e, s, h, _, _, _ = fake_batch(4)
    with torch.no_grad():
        p0 = m(x, e, s, h, torch.zeros(4, 1))
        direct = m.head(m.encoder(x, e, s, h)).unsqueeze(1)
    assert torch.allclose(p0, direct, atol=1e-6)
    print("ok  offset 0 returns the head applied to z(t0)")


def test_gradients_reach_dynamics_and_encoder():
    m = KANODEForecaster(22, 34, 10, h_in=6)
    x, e, s, h, offs, score, mask = fake_batch(8)
    forecast_loss(m(x, e, s, h, offs), score, mask).backward()
    g_f = sum(p.grad.abs().sum() for p in m.f.parameters()).item()
    g_enc = sum(p.grad.abs().sum() for p in m.encoder.convs.parameters()).item()
    assert g_f > 0 and g_enc > 0
    print(f"ok  gradients reach the KAN dynamics ({g_f:.2e}) and the conv encoder ({g_enc:.2e})")


def test_models_can_fit_small_batch():
    """Both forecasters must be able to memorise 32 samples x 4 horizons."""
    for cls in (KANODEForecaster, NoODEForecaster):
        m = cls(22, 34, 10, h_in=6)
        x, e, s, h, offs, score, mask = fake_batch(32)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        for _ in range(400):
            loss = forecast_loss(m(x, e, s, h, offs), score, mask, lam=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        mae = (m(x, e, s, h, offs)[..., :2] - score).abs().mean().item()
        assert mae < 0.5, (cls.__name__, mae)
        n = sum(p.numel() for p in m.parameters())
        print(f"ok  {cls.__name__} memorises a small batch (MAE {mae:.2f}, {n:,} parameters)")


if __name__ == "__main__":
    test_rk4_exact_solution()
    test_rk4_matches_torchdiffeq()
    test_offset_zero_is_initial_state()
    test_gradients_reach_dynamics_and_encoder()
    test_models_can_fit_small_batch()
    print("all tests passed")
