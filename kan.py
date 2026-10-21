"""
KAN building blocks used by Interpretable-MTLNet.

KANLinear  = paper Eq. (1)-(2):  y_j = sum_i phi_{j,i}(x_i) + b_j
             phi_{j,i}(x) = sum_k c_{j,i,k} B_k(x) + w_{j,i} * SiLU(x)
             B_k are cubic B-spline basis functions on a fixed grid (KAN paper, Liu et al. 2025)

KANConv1d  = paper Eq. (3):  H_{c,t} = sum_i sum_j phi_{c,i,j}(X_{t+j,i})
             i.e. slide a window of k days, flatten (features x offsets), apply KANLinear.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-3.0, 3.0), noise_scale=0.1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order

        # knot vector: G intervals inside grid_range, plus k extra knots on each side
        h = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())

        n_basis = grid_size + spline_order                       # number of B_k per edge
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))          # w_{j,i}
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_basis))  # c_{j,i,k}
        self.bias = nn.Parameter(torch.zeros(out_features))                              # b_j

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=noise_scale / math.sqrt(in_features))

    def b_splines(self, x):
        """Cox-de Boor recursion. x: (N, in) -> (N, in, grid_size + spline_order)"""
        g = self.grid                                            # (in, n_knots)
        x = x.unsqueeze(-1)                                      # (N, in, 1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)  # order 0
        for p in range(1, self.spline_order + 1):
            left = (x - g[:, : -(p + 1)]) / (g[:, p:-1] - g[:, : -(p + 1)])
            right = (g[:, p + 1:] - x) / (g[:, p + 1:] - g[:, 1:-p])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features)
        base = F.linear(F.silu(x), self.base_weight)
        spline = F.linear(self.b_splines(x).reshape(x.size(0), -1),
                          self.spline_weight.reshape(self.out_features, -1))
        out = base + spline + self.bias
        return out.reshape(*shape[:-1], self.out_features)

    @torch.no_grad()
    def edge_curve(self, j, i, xs):
        """phi_{j,i}(x) evaluated on xs (1-D tensor): the learned curve on one edge."""
        x = torch.zeros(len(xs), self.in_features, device=xs.device)
        x[:, i] = xs
        b = self.b_splines(x)[:, i, :]                           # (len, n_basis)
        return b @ self.spline_weight[j, i] + self.base_weight[j, i] * F.silu(xs)


class KANConv1d(nn.Module):
    """Temporal KAN convolution, 'valid' padding (no future days are ever created)."""
    def __init__(self, in_channels, out_channels, kernel_size, **kan_kwargs):
        super().__init__()
        self.k = kernel_size
        self.in_channels = in_channels
        self.kan = KANLinear(in_channels * kernel_size, out_channels, **kan_kwargs)

    def forward(self, x):                    # x: (B, T, D)
        win = x.unfold(1, self.k, 1)         # (B, T-k+1, D, k): day offsets j = 0..k-1
        win = win.reshape(x.size(0), win.size(1), -1)   # flatten (feature i, offset j)
        return self.kan(win)                 # (B, T-k+1, C)
