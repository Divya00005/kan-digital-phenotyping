"""Unit tests for the KAN building blocks. Run: python test_kan.py"""
import math
import torch
from kan import KANLinear, KANConv1d
from model import KANMTLNet, multitask_loss

torch.manual_seed(0)


def test_partition_of_unity():
    """Inside the grid, cubic B-spline bases must add up to exactly 1."""
    layer = KANLinear(1, 1)
    x = torch.linspace(-2.99, 2.99, 500).unsqueeze(1)
    s = layer.b_splines(x).sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), s.min()
    print("ok  B-spline bases sum to 1 inside the grid")


def test_fits_1d_curve():
    """A single KAN edge should learn sin(x) almost perfectly."""
    layer = KANLinear(1, 1, grid_size=10)
    x = torch.linspace(-3, 3, 400).unsqueeze(1)
    y = torch.sin(2 * x)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    for _ in range(1500):
        loss = ((layer(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 1e-3, loss.item()
    print(f"ok  one KAN edge fits sin(2x): MSE {loss.item():.2e}")


def test_conv_matches_equation_3():
    """Vectorised KANConv1d must equal a slow loop over (feature i, offset j)."""
    conv = KANConv1d(in_channels=3, out_channels=2, kernel_size=4)
    x = torch.randn(2, 10, 3)
    fast = conv(x)
    slow = torch.zeros_like(fast)
    for b in range(2):
        for t in range(10 - 4 + 1):
            win = x[b, t:t + 4, :].T.reshape(-1)          # order (feature i, offset j)
            slow[b, t] = conv.kan(win.unsqueeze(0))[0]
    assert torch.allclose(fast, slow, atol=1e-5)
    print("ok  KAN conv equals the loop version of Eq. 3")


def test_edge_decomposition():
    """Output = sum of per-edge curves + bias (Eq. 1): the basis of interpretability."""
    layer = KANLinear(3, 2)
    x = torch.randn(5, 3)
    manual = torch.stack([sum(layer.edge_curve(j, i, x[:, i]) for i in range(3)) + layer.bias[j]
                          for j in range(2)], dim=1)
    assert torch.allclose(layer(x), manual, atol=1e-5)
    print("ok  layer output = sum of its learned edge curves + bias")


def test_model_overfits_small_batch():
    """The full model must be able to memorise 32 samples (wiring + gradients work)."""
    m = KANMTLNet(d_in=22, e_in=34, s_in=10)
    X, E, S = torch.randn(32, 60, 22), torch.randn(32, 34), torch.randn(32, 10)
    Y = (torch.rand(32, 2) > 0.5).float()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(300):
        loss = multitask_loss(m(X, E, S), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = ((torch.sigmoid(m(X, E, S)) > 0.5).float() == Y).float().mean().item()
    assert acc > 0.95, acc
    print(f"ok  full model memorises 32 samples (accuracy {acc:.2f})")


if __name__ == "__main__":
    test_partition_of_unity()
    test_fits_1d_curve()
    test_conv_matches_equation_3()
    test_edge_decomposition()
    test_model_overfits_small_batch()
    print("all tests passed")
