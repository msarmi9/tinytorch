"""
Module for high-level containers.
"""

class Sequential:
    """Container for a feedforward neural net."""

    def __init__(self, layers, criterion):
        """Initialise layers and loss criterion."""
        self.layers = layers
        self.criterion = criterion

    def forward(self, x):
        """Pass a mini-batch through the network."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self):
        """Backpropagate gradients to the start of the network."""
        grad = self.criterion.backward()
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
