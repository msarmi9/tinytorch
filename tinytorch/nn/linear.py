"""
Module for linear layers.
"""

import numpy as np


class Linear:
    """Container for the forward and backward pass of a linear layer."""

    def __init__(self, n_inp, n_out):
        """Initialise layer with random weights and zero bias."""
        k = 1 / np.sqrt(n_inp)
        self.weights = np.random.uniform(-k, k, (n_inp, n_out))
        self.bias = np.zeros(n_out)

    def forward(self, x):
        """Pass a mini-batch through a linear layer."""
        self.x = x
        return x @ self.weights + self.bias

    def backward(self, grad):
        """Backpropagate the gradient given the preceding gradient."""
        self.grad_w = (self.x[:, :, None] @ grad[:, None, :]).mean(axis=0)
        self.grad_b = grad.mean(axis=0)
        return grad @ self.weights.T
