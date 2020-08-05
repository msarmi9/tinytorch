"""
Module for training utils.
"""

class Trainer:
    """Container for training a feedforward neural net."""

    def __init__(self, model, optimizer, train_dl, val_dl, metric):
        self.model = model
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.metric = metric

    def train_one_epoch(self):
        """Train for one epoch and return the loss."""
        loss, n = 0, 0
        for x, y in self.train_dl:
            y_hat = self.model.forward(x)
            batch_loss = self.model.criterion.forward(y_hat, y).sum()
            self.model.backward()
            self.optimizer.step()
            loss += batch_loss
            n += len(y)
        return loss / n

    def train(self, n_epochs, log_level=1):
        """Train for several epochs."""
        for epoch in range(n_epochs):
            loss = self.train_one_epoch()
            val_loss, val_metric = self.evaluate(self.val_dl)
            if (epoch + 1) % log_level == 0:
                print(f"epoch= {epoch:2d} | loss= {loss:.3f} | "
                      f"val_loss= {val_loss:.3f} | val_metric= {val_metric:.3f}")

    def evaluate(self, dl):
        """Return loss and metric on validation or test set."""
        loss, n, metric = 0, 0, 0
        for x, y in dl:
            y_hat = self.model.forward(x)
            batch_loss = self.model.criterion.forward(y_hat, y).sum()
            batch_metric = self.metric(y_hat, y)
            metric += len(y) * batch_metric
            loss += batch_loss
            n += len(y)
        return loss / n, metric / n
