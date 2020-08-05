"""
Module for evaluation metrics.
"""

def accuracy(y_hat, y):
    """Compute accuracy given soft binary predictions."""
    y_pred = y_hat > 0.5
    return (y_pred == y).mean()
