"""
RQ3 quantile model for CQR (pre-registered in CLAUDE.md, 2026-10-22).

QuantileNoODEForecaster = the KAN no-ODE forecaster (same encoder and no-ODE head) whose head
outputs, per horizon, 8 values: for depression then anxiety, the 5%, 10%, 90% and 95% quantiles
of the future score. Trained with the pinball loss only. At prediction time the 4 quantiles of
each task are sorted (so they never cross); CQR's guarantee holds for any quantile estimates.
"""
import torch
import numpy as np
from forecast_model import NoODEForecaster, Head

QUANTILES = (0.05, 0.10, 0.90, 0.95)
N_Q = len(QUANTILES)


class QuantileNoODEForecaster(NoODEForecaster):
    quantile = True                                  # tells train_forecast.py to use the pinball loss

    def __init__(self, d_in, e_in, s_in, h_in=0, latent=16, width=32):
        super().__init__(d_in, e_in, s_in, h_in, latent, width)
        self.head = Head(latent, n_out=2 * N_Q)     # (B, n_horizons, 8)


def pinball_loss(pred, score, mask):
    """Mean pinball (quantile) loss over target entries. pred (B, H, 8), score (B, H, 2), mask (B, H)."""
    m = mask.bool()
    if m.sum() == 0:
        return pred.sum() * 0
    q = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device)
    p = pred[m].reshape(-1, 2, N_Q)                  # (n, task, quantile)
    u = score[m].unsqueeze(-1) - p
    return torch.maximum(q * u, (q - 1) * u).mean()


def sort_quantiles(pred):
    """numpy (n, H, 8) -> the same with each task's 4 quantiles sorted (no crossing)."""
    s = pred.shape
    return np.sort(pred.reshape(*s[:-1], 2, N_Q), axis=-1).reshape(s)


def quantile_index(task, q):
    """Column of task (0 = depression, 1 = anxiety) and quantile level q in the 8 outputs."""
    return task * N_Q + QUANTILES.index(q)


def quantile_val_metrics(pred, score, mask, horizons):
    """Monitoring only (not conformal): raw coverage and width of the 10-90% and 5-95% bands."""
    out = {}
    for lo, hi, name in [(0.10, 0.90, "80"), (0.05, 0.95, "90")]:
        cov, wid = [], []
        for k in range(len(horizons)):
            m = mask[:, k]
            for t in range(2):
                a, b = pred[m, k, quantile_index(t, lo)], pred[m, k, quantile_index(t, hi)]
                y = score[m, k, t]
                cov.append(np.mean((y >= a) & (y <= b))); wid.append(np.mean(np.clip(b, 0, 6) - np.clip(a, 0, 6)))
        out[f"cover_{name}"], out[f"width_{name}"] = float(np.mean(cov)), float(np.mean(wid))
    return out
