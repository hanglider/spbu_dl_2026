"""Задание 3. Реализации Momentum, RMSProp, Adam и AdamW.

Формулы совпадают с torch.optim (с параметрами по умолчанию), чтобы их можно было
сверить с эталоном в тестах. Описание идей алгоритмов в README.md.
"""

import torch
from torch.optim import Optimizer


class Momentum(Optimizer):
    """SGD с моментом: v = beta * v + g;  w -= lr * v."""

    def __init__(self, params, lr=1e-2, momentum=0.9):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["v"] = torch.zeros_like(p)
                v = state["v"]
                v.mul_(group["momentum"]).add_(p.grad)
                p.sub_(group["lr"] * v)


class RMSProp(Optimizer):
    """s = alpha * s + (1 - alpha) * g^2;  w -= lr * g / (sqrt(s) + eps)."""

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["s"] = torch.zeros_like(p)
                s = state["s"]
                s.mul_(group["alpha"]).add_((1 - group["alpha"]) * p.grad**2)
                p.sub_(group["lr"] * p.grad / (s.sqrt() + group["eps"]))


class Adam(Optimizer):
    """Момент (m) + RMSProp (v) с поправкой на смещение нулевой инициализации.

    weight_decay здесь классический: добавляется к градиенту (L2-регуляризация).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def _decay(self, p, group):
        """Добавка к градиенту и изменение весов до шага Adam (AdamW переопределяет)."""
        return p.grad + group["weight_decay"] * p

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                g = self._decay(p, group)
                m.mul_(beta1).add_((1 - beta1) * g)
                v.mul_(beta2).add_((1 - beta2) * g**2)
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                p.sub_(group["lr"] * m_hat / (v_hat.sqrt() + group["eps"]))


class AdamW(Adam):
    """Adam с «развязанным» weight decay: веса уменьшаются напрямую, а не через градиент."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def _decay(self, p, group):
        p.mul_(1 - group["lr"] * group["weight_decay"])
        return p.grad
