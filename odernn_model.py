"""
RQ1 variant C (pre-registered in CLAUDE.md, 2026-10-22): an ODE INSIDE the encoder.

Both models read the 61 days before the forecast, day k = t0 - 60 + k (k = 60 is t0):
    sensing x_k   for k = 0 .. 59   (22 channels; the last one is the valid-day flag)
    survey  s_k   for k = 0 .. 60   (PHQ-2 / 6, GAD-2 / 6) where a survey exists

ODERNNForecaster   (Rubanova et al. 2019, ODE-RNN):
    h starts at 0. For each day:  h <- RK4 step of dh/dt = KAN(h) over 1 day   (not before day 0)
                                  h <- GRU_x(x_k, h)   if day k has valid sensing
                                  h <- GRU_s(s_k, h)   if day k has a survey
    Missing days get no update: h only evolves through the ODE.

GRUDtForecaster    (control, "RNN-delta-t"):
    the same two GRU updates on the same days, but h stays fixed between updates; instead the
    time since the previous update (in weeks) is an extra GRU input. The same-sized KAN block is
    applied once to the final h (h <- h + KAN(h)), like the variant A no-ODE control.

After the sequence (identical in both): [h, engineered E, history H] -> KAN, static S -> KAN,
-> z(t0) (16) -> no-ODE head of variant A (horizon as an extra input) -> 4 outputs per horizon.
"""
import torch
import torch.nn as nn
from kan import KANLinear
from forecast_model import KANDynamics, Head, DAY

N_DAYS = 61          # t0-60 .. t0


def rk4_step(f, z, dt):
    k1 = f(z)
    k2 = f(z + 0.5 * dt * k1)
    k3 = f(z + 0.5 * dt * k2)
    k4 = f(z + dt * k3)
    return z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class _SequenceForecaster(nn.Module):
    uses_surveys = True          # tells train_forecast.py to pass the survey arrays
    ode = None

    def __init__(self, d_in, e_in, s_in, h_in, hidden=32, latent=16, width=32,
                 mix_hidden=64, static_out=8):
        super().__init__()
        extra = 0 if self.ode else 1                                 # delta-t input (control only)
        self.gru_x = nn.GRUCell(d_in + extra, hidden)
        self.gru_s = nn.GRUCell(2 + extra, hidden)
        self.f = KANDynamics(hidden, width)                         # dh/dt (ODE) or one-off block
        mix_in = hidden + e_in + h_in
        self.mix_norm = nn.LayerNorm(mix_in)
        self.shared = KANLinear(mix_in, mix_hidden)
        self.static = KANLinear(s_in, static_out)
        self.out_norm = nn.LayerNorm(mix_hidden + static_out)
        self.to_latent = KANLinear(mix_hidden + static_out, latent)
        self.g = KANDynamics(latent, width, extra_in=1)             # no-ODE head (as variant A)
        self.head = Head(latent)
        self.hidden = hidden

    def walk(self, x, surv, surv_mask):
        raise NotImplementedError

    def encode(self, x, e, s, h, surv, surv_mask):
        hT = self.walk(x, surv, surv_mask)
        u = torch.cat([self.shared(self.mix_norm(torch.cat([hT, e, h], 1))), self.static(s)], 1)
        return self.to_latent(self.out_norm(u))                       # z(t0)

    def forward(self, x, e, s, h, offset_days, surv, surv_mask):
        z0 = self.encode(x, e, s, h, surv, surv_mask)
        tau = (offset_days * DAY / 8.0).unsqueeze(-1)               # horizon feature
        z_rep = z0.unsqueeze(1).expand(-1, offset_days.size(1), -1)
        return self.head(self.g(torch.cat([z_rep, tau], dim=-1)))   # (B, n_h, 4)


class ODERNNForecaster(_SequenceForecaster):
    ode = True

    def walk(self, x, surv, surv_mask):
        B = x.size(0)
        valid = x[..., -1:] > 0.5                                   # (B, 60, 1)
        smask = surv_mask.unsqueeze(-1)                             # (B, 61, 1)
        hs = x.new_zeros(B, self.hidden)
        for k in range(N_DAYS):
            if k > 0:
                hs = rk4_step(self.f, hs, DAY)                      # evolve day k-1 -> day k
            if k < N_DAYS - 1:
                hs = torch.where(valid[:, k], self.gru_x(x[:, k], hs), hs)
            hs = torch.where(smask[:, k], self.gru_s(surv[:, k], hs), hs)
        return hs


class GRUDtForecaster(_SequenceForecaster):
    ode = False

    def walk(self, x, surv, surv_mask):
        B = x.size(0)
        valid = x[..., -1:] > 0.5
        smask = surv_mask.unsqueeze(-1)
        hs = x.new_zeros(B, self.hidden)
        last = x.new_zeros(B, 1)                                    # day of the previous update
        for k in range(N_DAYS):
            if k < N_DAYS - 1:
                dt = (k - last) * DAY                               # weeks since previous update
                hs = torch.where(valid[:, k], self.gru_x(torch.cat([x[:, k], dt], 1), hs), hs)
                last = torch.where(valid[:, k], torch.full_like(last, k), last)
            dt = (k - last) * DAY
            hs = torch.where(smask[:, k], self.gru_s(torch.cat([surv[:, k], dt], 1), hs), hs)
            last = torch.where(smask[:, k], torch.full_like(last, k), last)
        return hs + self.f(hs)                                      # same KAN block, applied once
