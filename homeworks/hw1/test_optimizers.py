"""Тесты для optimizers.py."""

import unittest

import torch
import torch.nn as nn

import optimizers as my

# пары (наш оптимизатор, эталон из torch.optim) с одинаковыми гиперпараметрами
CASES = {
    "Momentum": (lambda p: my.Momentum(p, lr=0.05, momentum=0.9), lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    "RMSProp": (lambda p: my.RMSProp(p, lr=0.01, alpha=0.9), lambda p: torch.optim.RMSprop(p, lr=0.01, alpha=0.9)),
    "Adam": (lambda p: my.Adam(p, lr=0.01), lambda p: torch.optim.Adam(p, lr=0.01)),
    "Adam+L2": (
        lambda p: my.Adam(p, lr=0.01, weight_decay=0.1),
        lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1),
    ),
    "AdamW": (
        lambda p: my.AdamW(p, lr=0.01, weight_decay=0.1),
        lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1),
    ),
}


def make_net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 1)).double()


def run(make_opt, steps=30):
    net = make_net()
    opt = make_opt(net.parameters())
    torch.manual_seed(1)
    x = torch.randn(64, 5, dtype=torch.float64)
    y = 3 * torch.tanh(x @ torch.randn(5, 1, dtype=torch.float64))  # цель, которую сеть может выучить
    losses = []
    for _ in range(steps):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return net, losses


class TestOptimizers(unittest.TestCase):
    def test_matches_torch(self):
        for name, (mine, ref) in CASES.items():
            with self.subTest(name):
                net, _ = run(mine)
                ref_net, _ = run(ref)
                for p, q in zip(net.parameters(), ref_net.parameters()):
                    torch.testing.assert_close(p, q)

    def test_decreases_loss(self):
        for name, (mine, _) in CASES.items():
            with self.subTest(name):
                _, losses = run(mine, steps=100)
                self.assertLess(losses[-1], 0.5 * losses[0])

    def test_converges_on_quadratic(self):
        # минимум f(w) = sum(a * (w - c)^2) в точке c, при плохой обусловленности (a от 1 до 100)
        a = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)
        c = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)
        settings = {
            my.Momentum: dict(lr=0.005),
            my.RMSProp: dict(lr=0.01),
            my.Adam: dict(lr=0.05),
            my.AdamW: dict(lr=0.05, weight_decay=0.0),
        }
        for cls, kwargs in settings.items():
            with self.subTest(cls.__name__):
                w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
                opt = cls([w], **kwargs)
                for _ in range(2000):
                    loss = (a * (w - c) ** 2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                torch.testing.assert_close(w.detach(), c, atol=1e-2, rtol=0)

    def test_first_adam_step_is_lr_times_sign(self):
        # из-за поправки на смещение первый шаг Adam равен lr * sign(g) независимо от масштаба g
        w = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        opt = my.Adam([w], lr=0.1)
        (w * torch.tensor([1000.0, -0.001, 5.0])).sum().backward()
        opt.step()
        torch.testing.assert_close(w.detach(), torch.tensor([0.9, 1.1, 0.9]))

    def test_adamw_decays_weights_without_gradient(self):
        # при нулевом градиенте AdamW только сжимает веса: w_t = w_0 * (1 - lr * wd)^t
        w = torch.tensor([2.0, -4.0], requires_grad=True)
        opt = my.AdamW([w], lr=0.1, weight_decay=0.5)
        for _ in range(10):
            w.grad = torch.zeros_like(w)
            opt.step()
        torch.testing.assert_close(w.detach(), torch.tensor([2.0, -4.0]) * 0.95**10)

    def test_params_without_grad_are_skipped(self):
        for name, (mine, _) in CASES.items():
            with self.subTest(name):
                w = torch.ones(3, requires_grad=True)
                frozen = torch.ones(3, requires_grad=True)
                opt = mine([w, frozen])
                w.sum().backward()
                opt.step()
                torch.testing.assert_close(frozen.detach(), torch.ones(3))
                self.assertFalse(torch.equal(w.detach(), torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
