"""Общий код для заданий 1 и 3: модель, данные FashionMNIST и цикл обучения."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST

DATA_DIR = Path(__file__).parent / "data"


class BaseNetwork(nn.Module):
    def __init__(self, act_fn, input_size=784, num_classes=10, hidden_sizes=(512, 256, 256, 128)):
        super().__init__()
        layers = []
        layer_sizes = [input_size] + list(hidden_sizes)
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]), act_fn]
        layers += [nn.Linear(layer_sizes[-1], num_classes)]
        self.layers = nn.ModuleList(layers)
        self.config = {
            "act_fn": act_fn.__class__.__name__,
            "input_size": input_size,
            "num_classes": num_classes,
            "hidden_sizes": list(hidden_sizes),
        }

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


def load_fashion_mnist():
    """Весь датасет сразу в тензоры: для MLP это быстрее, чем DataLoader."""
    out = []
    for train in (True, False):
        ds = FashionMNIST(DATA_DIR, train=train, download=True)
        x = ds.data.float().div(255).sub(0.2860).div(0.3530).view(-1, 784)
        out += [x, ds.targets]
    return out  # x_train, y_train, x_test, y_test


def batches(x, y, batch_size, shuffle=True):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), batch_size):
        b = idx[i : i + batch_size]
        yield x[b], y[b]


@torch.no_grad()
def accuracy(model, x, y, batch_size=4096):
    model.eval()
    correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in batches(x, y, batch_size, False))
    return correct / len(x)


def grad_norm(model):
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    return torch.cat(grads).norm().item()


def train(model, optimizer, data, epochs=3, batch_size=128):
    """Возвращает историю: loss и норма градиента на каждом шаге, точность на тесте по эпохам."""
    x_train, y_train, x_test, y_test = data
    history = {"loss": [], "grad_norm": [], "test_acc": []}
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(x_train, y_train, batch_size):
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            history["loss"].append(loss.item())
            history["grad_norm"].append(grad_norm(model))
            optimizer.step()
        history["test_acc"].append(accuracy(model, x_test, y_test))
    return history
