"""
Fixed-step Runge-Kutta 4 solver, written out so every step is visible.

For dz/dt = f(z) with step size dt:
    k1 = f(z)
    k2 = f(z + dt/2 * k1)
    k3 = f(z + dt/2 * k2)
    k4 = f(z + dt * k3)
    z_next = z + dt/6 * (k1 + 2 k2 + 2 k3 + k4)

Gradients flow through these operations with ordinary backpropagation
(no adjoint method needed for a model this small).
"""
import torch


def rk4_trajectory(f, z0, n_steps, dt):
    """Integrate from t=0 for n_steps of size dt. Returns states at every step:
    tensor (B, n_steps + 1, d), where index i is time i * dt."""
    states = [z0]
    z = z0
    for _ in range(n_steps):
        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(z)
    return torch.stack(states, dim=1)
