"""Тесты для autograd.Node. Эталонные градиенты берём из torch.autograd."""

import random
import unittest

import torch

from autograd import Node


def torch_grads(fn, *values):
    xs = [torch.tensor(float(v), dtype=torch.float64, requires_grad=True) for v in values]
    out = fn(*xs)
    out.backward()
    return out.item(), [x.grad.item() for x in xs]


def node_grads(fn, *values):
    xs = [Node(v) for v in values]
    out = fn(*xs)
    out.backward()
    return out.data, [x.grad for x in xs]


class TestNode(unittest.TestCase):
    def assert_matches_torch(self, node_fn, torch_fn, *values):
        out, grads = node_grads(node_fn, *values)
        ref_out, ref_grads = torch_grads(torch_fn, *values)
        self.assertAlmostEqual(out, ref_out)
        for g, ref in zip(grads, ref_grads):
            self.assertAlmostEqual(g, ref)

    def test_add(self):
        out, (ga, gb) = node_grads(lambda a, b: a + b, 2, 5)
        self.assertEqual((out, ga, gb), (7, 1, 1))

    def test_mul(self):
        out, (ga, gb) = node_grads(lambda a, b: a * b, 2, 5)
        self.assertEqual((out, ga, gb), (10, 5, 2))

    def test_relu(self):
        for x, expected_out, expected_grad in [(3, 3, 1), (-3, 0, 0), (0, 0, 0)]:
            out, (g,) = node_grads(lambda a: a.relu(), x)
            self.assertEqual((out, g), (expected_out, expected_grad))

    def test_python_numbers(self):
        out, (g,) = node_grads(lambda a: 2 * a + 1 + a * 3, 4)
        self.assertEqual((out, g), (21, 5))

    def test_example_from_task(self):
        # d = 2 + (-3) * 10 = -28 < 0, поэтому ReLU обнуляет градиент для всего, что ниже e
        a, b, c = Node(2), Node(-3), Node(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        self.assertEqual([n.data for n in (a, b, c, d, e)], [2, -3, 10, -28, 0])
        self.assertEqual([n.grad for n in (a, b, c, d, e)], [0, 0, 0, 0, 1])

    def test_positive_branch(self):
        self.assert_matches_torch(
            lambda a, b, c: (a + b * c).relu(),
            lambda a, b, c: torch.relu(a + b * c),
            2, 3, 10,
        )

    def test_node_used_twice(self):
        # градиенты от двух использований одного узла должны складываться
        self.assert_matches_torch(lambda a: a * a + a, lambda a: a * a + a, 3)

    def test_backward_through_shared_subgraph(self):
        def f(a, b):
            c = a * b
            return (c + c * a).relu() * c

        def f_torch(a, b):
            c = a * b
            return torch.relu(c + c * a) * c

        self.assert_matches_torch(f, f_torch, 1.5, -2.0)
        self.assert_matches_torch(f, f_torch, 1.5, 2.0)

    def test_random_expressions(self):
        rng = random.Random(0)
        for _ in range(50):
            vals = [rng.uniform(-3, 3) for _ in range(4)]

            def f(a, b, c, d, relu=lambda n: n.relu()):
                h1 = relu(a * b + c)
                h2 = relu(h1 * d + a)
                return h1 * h2 + b * d

            self.assert_matches_torch(f, lambda *xs: f(*xs, relu=torch.relu), *vals)


if __name__ == "__main__":
    unittest.main()
