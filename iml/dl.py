from typing import Any
import numpy as np
from dataclasses import dataclass
from math import sqrt
from numpy.random import randn, uniform
from functools import cache
from torch.utils.data import DataLoader

class Module:
    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def backward(self, *args, **kwargs) -> Any:
        return None
        # implemented so we don't have to declare a backward method
        # for something like an embedding layer


class Sequential(Module):
    def __init__(self, *components: Module):
        self.components: tuple[Module, ...] = components

    def forward(self, x):
        for c in self.components:
            x = c(x)
        return x

    def backward(self, dy):
        for c in reversed(self.components):
            dy = c.backward(dy)
        return dy


@dataclass
class Linear(Module):
    """
    A linear module that adjusts itself using gradient descent
    """
    d_in: int
    d_out: int
    lr: float = 0.01  # learning rate
    init_scale: float = 1

    def __post_init__(self):
        bound = sqrt(6 / (self.d_in + self.d_out))  # Glorot initialization
        bound *= self.init_scale
        self.weights = uniform(-bound, bound, (self.d_in, self.d_out))
        self.biases = np.zeros(self.d_out)

    def forward(self, x) -> np.ndarray:
        # x: b d_in
        self.x = x.copy()
        return x @ self.weights + self.biases

    def backward(self, gy) -> np.ndarray:
        # x: b d_in
        # gy: b d_out (read: "gradient at y")
        if not hasattr(self, 'x'):
            raise ValueError('You need to call forward() before you call backward()')

        # just vanilla gradient descent for the objective L
        # assuming that gy is dL/dy
        gx = gy @ self.weights.T
        self.weights -= self.lr * self.x.T @ gy
        self.biases -= self.lr * gy.sum(axis=0)
        return gx


@dataclass
class Scaler(Module):
    def __init__(self, mean=None, var=None):
        if mean is None:
            mean = np.array(0)
        self.mean = mean

        if var is None:
            var = np.array(1)
        self.var = var

    def forward(self, x):
        return (x - self.mean) / np.sqrt(self.var)

    def backward(self, dy):
        return dy / np.sqrt(self.var)

    @staticmethod
    def fit(x: np.ndarray):
        return Scaler(mean=x.mean(axis=0), var=x.var(axis=0))


@cache
def _mnist():
    import datasets
    print("Loading mnist dataset (will be cached...)")
    return datasets.load_dataset('mnist').with_format("torch")


def mnist_iter(split="train", batch_size=1024):
    """
    Return an iterator into either the train or the test split of the MNIST
    dataset. Each value returned by the iterator is a tuple (x, y) where x
    is a batch of images and y is a batch of labels.
    """
    loader = DataLoader(_mnist()[split], batch_size=batch_size)  # pyright: ignore
    def transform(batch):
        x = batch["image"].reshape(-1, 28 * 28).numpy() / 255
        y = batch["label"].numpy()
        return x, y

    return map(transform, loader)
