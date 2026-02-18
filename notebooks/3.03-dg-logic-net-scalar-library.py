# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append("..")

# %%
import math

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lenet5.data.logic as logic_data
import lenet5.modeling.autograd as ag
from lenet5.modeling.autograd import Value
import lenet5.modeling.logic_net_3_03 as logic_net
from lenet5.modeling.logic_net_3_03 import to_logic, predict, result_table, calc_loss, calc_accuracy, mse_loss, train_epoch

# %%
class Hyper:
    def __init__(self, name, seed, features, targets, model, optimizer, lr, nepochs):
        self.name = name
        self.seed = seed
        self.features = features
        self.targets = targets
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.nepochs = nepochs

class Result:
    def __init__(self, hyper: Hyper, model, epochs, losses, accuracies, result_table):
        self.hyper = hyper
        self.model = model
        self.epochs = epochs.copy()
        self.losses = losses.copy()
        self.accuracies = accuracies.copy()
        self.result_table = result_table.copy()

# %%
seed = 12345
features = logic_data.features
targets = logic_data.targets_xor
model = logic_net.TwoLayer
nepochs = 10000
scenarios = [
    Hyper(name="s0", seed=seed, features=features, targets=targets, model=model, optimizer=ag.Adam, lr=0.01, nepochs=nepochs),
    Hyper(name="s1", seed=420083464, features=features, targets=targets, model=model, optimizer=ag.Adam, lr=0.01, nepochs=nepochs),
    Hyper(name="s2", seed=555137262, features=features, targets=targets, model=model, optimizer=ag.Adam, lr=0.01, nepochs=nepochs),
    Hyper(name="s3", seed=605492662, features=features, targets=targets, model=model, optimizer=ag.Adam, lr=0.01, nepochs=nepochs),
    Hyper(name="s4", seed=713831756, features=features, targets=targets, model=model, optimizer=ag.Adam, lr=0.01, nepochs=nepochs),
]

# %%
def run_scenario(params: Hyper) -> Result:
    print(f"Running scenario: seed={params.seed}, model={params.model.__name__}, optimizer={params.optimizer.__name__}, lr={params.lr}, nepochs={params.nepochs}")
    def rand(): return np.random.normal(loc=0.5, scale=0.5)
    np.random.seed(params.seed)
    model = params.model(rand)
    optimizer = params.optimizer(model.parameters(), lr=params.lr)

    epochs = []
    losses = []
    accuracies = []

    for epoch in tqdm(range(params.nepochs)):
        optimizer.zero_grad()
        train_epoch(model, mse_loss, features, targets)
        optimizer.step()

        pred = predict(model, params.features)
        loss = calc_loss(mse_loss, pred, params.targets)
        acc = calc_accuracy(pred, params.targets)
        epochs.append(epoch)
        losses.append(loss)
        accuracies.append(acc)

    rt = result_table(params.features, pred, params.targets)
    return Result(hyper=params, model=model, epochs=epochs, losses=losses, accuracies=accuracies, result_table=rt)

results = []
for scenario in scenarios:
    results.append(run_scenario(scenario))

# %%
fig, axes = plt.subplots()
for result in results:
    axes.plot(result.epochs, result.losses, label=result.hyper.name)
axes.set_xscale("log")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.set_title("Loss vs Epoch")
axes.legend()
plt.show()

# %%
results[2].result_table

# %%
results[3].result_table

# %%
results[2].model.parameters()

# %%
results[3].model.parameters()

# %%
