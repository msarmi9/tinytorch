"""
Module for data loading utils.
"""

import numpy as np


class Dataset:
    """Container for returning inputs and targets."""

    def __init__(self, X, y):
        """Initialise inputs and re-shape targets as a column vector."""
        self.X = X
        self.y = y.reshape(-1, 1)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __setitem__(self, idx, val):
        self.X[idx], self.y[idx] = val

    def __len__(self):
        return len(self.y)


class DataLoader:
    """Container for returning a mini-batch of inputs and targets."""

    def __init__(self, ds, batch_size, shuffle=False):
        """Initialise dataset and batch size."""
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle

    def shuffle_data(self):
        """Shuffle inputs and targets."""
        idxs = np.random.permutation(len(self.ds))
        self.ds = Dataset(*self.ds[idxs])

    def __iter__(self):
        """Yield a mini-batch of inputs and targets."""
        if self.shuffle:
            self.shuffle_data()
        n_batches = len(self.ds) // self.batch_size
        for i in range(n_batches):
            yield self.ds[i * self.batch_size: (i + 1) * self.batch_size]
