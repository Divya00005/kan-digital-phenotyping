"""
Interpretable-MTLNet (Ali, Ni & Kundur, 2025) re-implemented from the paper.

   daily series (B, 60, D) --> KANConv1d k=3 ┐
                           --> KANConv1d k=5 ├-> average over time -> concat (64) ┐
                           --> KANConv1d k=7 ┘                                     ├-> shared KAN -> u (64)
   engineered features (B, E) ------------------------------------------------------┘
   static covariates   (B, S) --> static KAN -> s (8)
   [u ; s] --> depression KAN head -> logit
           --> anxiety    KAN head -> logit

Choices the paper leaves open (documented):
 * 64 channels in total across the three scales (Appendix B), split 22/21/21
 * grid size G=5, spline order k=3 (KAN paper defaults), grid range [-3, 3]
 * LayerNorm before the shared and head KAN layers keeps hidden values inside the spline grid
"""
import torch
import torch.nn as nn
from kan import KANLinear, KANConv1d


class KANMTLNet(nn.Module):
    def __init__(self, d_in, e_in, s_in, kernels=(3, 5, 7), total_channels=64,
                 hidden=64, static_out=8, n_tasks=2, use_engineered=True,
                 use_static=True, grid_size=5, spline_order=3):
        super().__init__()
        kw = dict(grid_size=grid_size, spline_order=spline_order)
        per = [total_channels // len(kernels)] * len(kernels)
        per[0] += total_channels - sum(per)                       # 22, 21, 21
        self.convs = nn.ModuleList(KANConv1d(d_in, c, k, **kw) for c, k in zip(per, kernels))
        self.use_engineered, self.use_static = use_engineered, use_static

        mix_in = total_channels + (e_in if use_engineered else 0)
        self.mix_norm = nn.LayerNorm(mix_in)
        self.shared = KANLinear(mix_in, hidden, **kw)             # shared KAN mixing

        head_in = hidden
        if use_static:
            self.static = KANLinear(s_in, static_out, **kw)       # static KAN processor
            head_in += static_out
        self.head_norm = nn.LayerNorm(head_in)
        self.heads = nn.ModuleList(KANLinear(head_in, 1, **kw) for _ in range(n_tasks))

    def forward(self, x, e, s):
        pooled = [conv(x).mean(dim=1) for conv in self.convs]     # global average pooling
        z = torch.cat(pooled, dim=1)
        if self.use_engineered:
            z = torch.cat([z, e], dim=1)
        u = self.shared(self.mix_norm(z))
        if self.use_static:
            u = torch.cat([u, self.static(s)], dim=1)
        u = self.head_norm(u)
        return torch.cat([h(u) for h in self.heads], dim=1)        # logits (B, n_tasks)


def focal_loss(logits, y, alpha=0.25, gamma=2.0):
    """Paper Eq. (7), averaged over samples."""
    p = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = p * y + (1 - p) * (1 - y)
    a_t = alpha * y + (1 - alpha) * (1 - y)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def multitask_loss(logits, y, weights=(1.2, 0.8), alpha=0.25, gamma=2.0):
    """Paper Eq. (6): weighted sum of per-task focal losses."""
    return sum(w * focal_loss(logits[:, t], y[:, t], alpha, gamma)
               for t, w in enumerate(weights))
