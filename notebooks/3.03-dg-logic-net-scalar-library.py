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
import lenet5.modeling.logic_net_03 as logic_net

# %%
def to_logic(x: float) -> bool:
    return x > 0.5

def predict(model, features):
    result = []
    for i in range(features.shape[0]):
        pred = Value.unwrap(model.forward(features[i,0].item(), features[i,1].item()))
        result.append(pred)
    return result

def result_table(features, pred: list, targets: np.array):
    df = pd.DataFrame(features, columns=["in1", "in2"])
    df["pred_raw"] = pred
    df["pred"] = [to_logic(p) for p in pred]
    df["target"] = pd.Series(targets.flatten()).map(to_logic)
    return df

def calc_loss(pred: list, target: np.array):
    loss = 0.
    for i in range(len(pred)):
        loss += (pred[i] - target[i].item()) ** 2
    return loss / len(pred)

def calc_accuracy(pred: list, target: np.array):
    correct = 0
    for i in range(len(pred)):
        if to_logic(pred[i]) == to_logic(target[i].item()):
            correct += 1
    return correct / len(pred)

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
nepochs = 30000
scenarios = [
    Hyper(name="Adam", seed=seed, features=features, targets=targets, model=model, optimizer=ag.Adam, lr=0.01, nepochs=nepochs),
    Hyper(name="AdamW", seed=seed, features=features, targets=targets, model=model, optimizer=ag.AdamW, lr=0.01, nepochs=nepochs),
]

# %%
def run_scenario(params: Hyper) -> Result:
    print(f"Running scenario: seed={params.seed}, model={params.model.__name__}, optimizer={params.optimizer.__name__}, lr={params.lr}, nepochs={params.nepochs}")
    generator = np.random.default_rng(params.seed)
    model = params.model(generator)
    optimizer = params.optimizer(model.parameters(), lr=params.lr)

    epochs = []
    losses = []
    accuracies = []

    for epoch in tqdm(range(params.nepochs)):
        optimizer.zero_grad()
        for i in range(params.features.shape[0]):
            pred = model.forward(params.features[i,0].item(), params.features[i,1].item())
            loss = (pred - params.targets[i].item()) ** 2
            loss.backward()
        optimizer.step()
        pred = predict(model, params.features)
        loss = calc_loss(pred, params.targets)
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
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.set_title("Loss vs Epoch")
axes.legend()
plt.show()

# %%
results[0].result_table

# %%
results[1].result_table

# %%
results[0].model.parameters()

# %%
results[1].model.parameters()
