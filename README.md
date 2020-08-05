## Tinytorch: A Deep Learning Library Written in Numpy

Welcome to tinytorch! Tinytorch is a small proof-of-concept library for those interested in learning how to write their own deep learning library in numpy. Naturally, it isn't meant for production purposes. Rather, tinytorch means to serve as a guide for those who want to really sink their teeth into the internals of neural nets.  


## What Can Tinytorch Do?

Tinytorch is under active development and new features will be added periodically. At the current moment, tinytorch offers  

- [x] Datasets & Dataloaders
- [x] Feedforward networks
- [ ] Convolutional neural nets
- [ ] Recurrent neural nets
- [ ] GPU support via `numba`


## Quick-start Guide

Here's a quick example of how to train a two-layer binary classifier with `tinytorch`.

```python
import tinytorch as tt
import tinytorch.nn as nn

# Initialise layers and criterion
criterion = nn.BinaryCrossEntropy()
layers = [nn.Linear(n_inp, 20), nn.ReLU(), nn.Linear(20, 1), nn.Sigmoid()]
model = nn.Sequential(layers, criterion)

# Initialise optimizer and trainer
optimizer = nn.SGD(model, lr=0.1)
trainer = tt.Trainer(model, optimizer, train_dl, val_dl, metric=tt.accuracy)
trainer.train(10)
```


## Installation Guide

To set up `tinytorch` for local development, clone the repo and run

```
% pip install -e .
```
