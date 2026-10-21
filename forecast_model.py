"""
RQ1 forecasting models.

Both models share the same encoder (the reproduced KAN-MTLNet front-end):
    60-day sensing -> multi-scale KAN conv -> pooled  ┐
    engineered features --------------------------------├-> shared KAN -> [ ; static KAN] -> z(t0)
    history-score features (input version B only) ------┘

KANODEForecaster  (variant A, KAN-ODE paper + Latent ODE paper):
    dz/dt = KAN(z)  integrated with RK4 in 1-day steps from t0 to t0 + offset
    heads read z(t0 + offset)

NoODEForecaster  (control):
    the same KAN capacity applied ONCE to [z(t0), offset]: a direct mapping, no integration

Time unit: weeks. One RK4 step = 1 day = 1/7 week, so every real target day is an exact step.
Head outputs per horizon: [depression score, anxiety score, depression logit, anxiety logit].
"""
import torch
import torch.nn as nn
from kan import KANLinear, KANConv1d
from model import focal_loss
from ode import rk4_trajectory

DAY = 1.0 / 7.0          # one day in weeks
MAX_DAYS = 8 * 7 + 3     # longest horizon + tolerance


class Encoder(nn.Module):
    def __init__(self, d_in, e_in, s_in, h_in=0, latent=16, kernels=(3, 5, 7),
                 total_channels=64, hidden=64, static_out=8):
        super().__init__()
        per = [total_channels // len(kernels)] * len(kernels)
        per[0] += total_channels - sum(per)
        self.convs = nn.ModuleList(KANConv1d(d_in, c, k) for c, k in zip(per, kernels))
        self.use_hist = h_in > 0
        mix_in = total_channels + e_in + h_in
        self.mix_norm = nn.LayerNorm(mix_in)
        self.shared = KANLinear(mix_in, hidden)
        self.static = KANLinear(s_in, static_out)
        self.out_norm = nn.LayerNorm(hidden + static_out)
        self.to_latent = KANLinear(hidden + static_out, latent)

    def forward(self, x, e, s, h=None):
        z = torch.cat([conv(x).mean(1) for conv in self.convs] + [e], dim=1)
        if self.use_hist:
            z = torch.cat([z, h], dim=1)
        u = torch.cat([self.shared(self.mix_norm(z)), self.static(s)], dim=1)
        return self.to_latent(self.out_norm(u))                       # z(t0)


class KANDynamics(nn.Module):
    """f(z) = dz/dt. Inputs to each KAN layer squashed to [-1, 1] with tanh (KAN-ODE paper)."""
    def __init__(self, latent=16, width=32, extra_in=0):
        super().__init__()
        self.l1 = KANLinear(latent + extra_in, width, grid_range=(-1.0, 1.0))
        self.l2 = KANLinear(width, latent, grid_range=(-1.0, 1.0))

    def forward(self, z):
        return self.l2(torch.tanh(self.l1(torch.tanh(z))))


class Head(nn.Module):
    def __init__(self, latent=16, n_out=4):
        super().__init__()
        self.norm = nn.LayerNorm(latent)
        self.kan = KANLinear(latent, n_out)

    def forward(self, z):
        return self.kan(self.norm(z))


class KANODEForecaster(nn.Module):
    def __init__(self, d_in, e_in, s_in, h_in=0, latent=16, width=32):
        super().__init__()
        self.encoder = Encoder(d_in, e_in, s_in, h_in, latent)
        self.f = KANDynamics(latent, width)
        self.head = Head(latent)

    def forward(self, x, e, s, h, offset_days):
        z0 = self.encoder(x, e, s, h)
        traj = rk4_trajectory(self.f, z0, MAX_DAYS, DAY)               # (B, MAX_DAYS+1, latent)
        idx = offset_days.long().clamp(0, MAX_DAYS)                     # (B, n_h)
        z_t = torch.gather(traj, 1, idx.unsqueeze(-1).expand(-1, -1, traj.size(2)))
        return self.head(z_t)                                           # (B, n_h, 4)


class NoODEForecaster(nn.Module):
    """Control: same encoder, same KAN sizes as the dynamics, but no integration.
    The horizon (in weeks / 8) is given as an extra input."""
    def __init__(self, d_in, e_in, s_in, h_in=0, latent=16, width=32):
        super().__init__()
        self.encoder = Encoder(d_in, e_in, s_in, h_in, latent)
        self.g = KANDynamics(latent, width, extra_in=1)
        self.head = Head(latent)

    def forward(self, x, e, s, h, offset_days):
        z0 = self.encoder(x, e, s, h)
        n_h = offset_days.size(1)
        tau = (offset_days * DAY / 8.0).unsqueeze(-1)                   # horizon feature
        z_rep = z0.unsqueeze(1).expand(-1, n_h, -1)
        return self.head(self.g(torch.cat([z_rep, tau], dim=-1)))


def forecast_loss(pred, score, mask, lam=10.0, task_w=(1.2, 0.8), cutoff=3.0):
    """Masked loss over horizons: score MSE + lam * weighted focal loss on screen status."""
    m = mask.bool()
    if m.sum() == 0:
        return pred.sum() * 0
    p, y = pred[m], score[m]                                            # (n, 4), (n, 2)
    mse = ((p[:, :2] - y) ** 2).mean()
    lab = (y >= cutoff).float()
    focal = task_w[0] * focal_loss(p[:, 2], lab[:, 0]) + task_w[1] * focal_loss(p[:, 3], lab[:, 1])
    return mse + lam * focal
