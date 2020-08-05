"""
Module for gradient descent optimizers.
"""

import tinytorch.nn as nn


class SGD:
    """Container for updating a model's weights via SGD."""

    def __init__(self, model, lr):
        """Initialise model parameters and learning rate."""
        self.model = model
        self.lr = lr

    def step(self):
        """Update weights and biases of all linear layers."""
        for layer in self.model.layers:
            if isinstance(layer, nn.Linear):
                layer.weights -= self.lr * layer.grad_w
                layer.bias -= self.lr * layer.grad_b
