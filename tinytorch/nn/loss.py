"""
Module for loss criterions.
"""

import numpy as np


class BinaryCrossEntropy:
    """Container for the forward and backward pass of BCE."""

    def forward(self, y_hat, y):
        """Return binary cross entropy given predictions and targets."""
        self.y_hat, self.y = y_hat.clip(min=1e-8, max=1 - 1e-8), y
        return -np.where(y == 1, np.log(self.y_hat), np.log(1 - self.y_hat))

    def backward(self):
        """Backpropagate the gradient with respect to predictions."""
        return (self.y_hat - self.y) / (self.y_hat * (1 - self.y_hat))
