"""
Module for activaiton functions.
"""

import numpy as np


class ReLU:
    """Container for the forward and backward pass of ReLU."""

    def forward(self, x):
        """Pass a mini-batch through ReLU."""
        self.x = x
        return np.where(x > 0, x, 0)

    def backward(self, grad):
        """Return the gradient where x is positive, otherwise zero."""
        return np.where(self.x > 0, grad, 0)


class Sigmoid:
    """Container for the forward and backward pass of sigmoid."""

    def forward(self, x):
        """Pass a mini-batch through a sigmoid layer."""
        self.y_hat = np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return self.y_hat

    def backward(self, grad):
        """Backpropagate the gradient given the preceding gradient."""
        return self.y_hat * (1 - self.y_hat) * grad
