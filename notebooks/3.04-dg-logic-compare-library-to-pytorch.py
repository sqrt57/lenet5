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
import collections

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lenet5.data.logic as logic_data
import lenet5.modeling.autograd as ag
from lenet5.modeling.autograd import Value
import lenet5.modeling.logic_net_3_03 as logic_net
from lenet5.modeling.logic_net_3_03 import to_logic, predict, result_table, calc_loss, calc_accuracy, mse_loss, train_epoch

import torch
from torch import nn
import torch.nn.functional as F


# %% [markdown]
# ## Common initialization

# %%
def rand():
    return np.random.normal(loc=0.5, scale=0.5)

features = logic_data.features
targets = logic_data.targets_xor

# %%
features.dtype


# %% [markdown]
# ## Equivalent Pytorch model

# %%
class TwoLayerTorch(nn.Module):
    def __init__(self, rand):
        super().__init__()
        self.weight1 = nn.Parameter(torch.tensor([[rand(), rand()], [rand(), rand()]], dtype=torch.float64))
        self.bias1 = nn.Parameter(torch.zeros((2,), dtype=torch.float64))
        self.weight2 = nn.Parameter(torch.tensor([[rand()], [rand()]], dtype=torch.float64))
        self.bias2 = nn.Parameter(torch.zeros((1,), dtype=torch.float64))

    def activation(x):
        return torch.tanh(2*(x-0.5)) / (2 * math.tanh(2 * 0.5)) + 0.5
    
    def forward(self, x):
        y1 = TwoLayerTorch.activation(x @ self.weight1 + self.bias1)
        y2 = TwoLayerTorch.activation(y1 @ self.weight2 + self.bias2)
        return y2


# %% [markdown]
# ## Simple forward and backward

# %%
np.random.seed(12345)
model1 = logic_net.TwoLayer(rand)
print(model1.parameters())
print(model1.forward(0.3, 0.7))

print("----")
print(model1.w111.grad)
model1.forward(0.3, 0.7).backward()
print(model1.w111.grad)
model1.forward(0.3, 0.7).backward()
print(model1.w111.grad)

# %%
np.random.seed(12345)
model2 = TwoLayerTorch(rand)
print(list(model2.parameters()))
print(model2(torch.tensor([[0.3, 0.7]], dtype=torch.float64)))

print("----")
print(model2.weight1.grad)
model2(torch.tensor([[0.3, 0.7]], dtype=torch.float64)).backward()
print(model2.weight1.grad)
model2(torch.tensor([[0.3, 0.7]], dtype=torch.float64)).backward()
print(model2.weight1.grad)

# %% [markdown]
# ## Optimization loops

# %%
Hyper = collections.namedtuple('Hyper', 'name seed optimizer lr nepochs')
Result = collections.namedtuple('Result', 'hyper model epochs losses accuracies result_table')

# %%
ag_optimizers = { 'SGD': ag.SGD, 'Adam': ag.Adam, 'AdamW': ag.AdamW, }
pytorch_optimizers = { 'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW, }


# %%
def result_table_torch(features, pred, targets):
    df = pd.DataFrame(features, columns=["in1", "in2"])
    df["pred_raw"] = pred.detach().numpy()
    df["pred"] = to_logic(pred)
    df["target"] = pd.Series(targets.flatten()).map(to_logic)
    return df


# %%
def run_ag(params: Hyper):
    print(f"Running AG scenario: seed={params.seed}, optimizer={params.optimizer}, lr={params.lr}, nepochs={params.nepochs}")
    np.random.seed(params.seed)
    model = logic_net.TwoLayer(rand)
    optimizer = ag_optimizers[params.optimizer](model.parameters(), lr=params.lr)

    epochs = []
    losses = []
    accuracies = []

    pred = predict(model, features)
    loss = calc_loss(mse_loss, pred, targets).item()
    acc = calc_accuracy(pred, targets)
    epochs.append(0)
    losses.append(loss)
    accuracies.append(acc)

    for epoch in tqdm(range(params.nepochs)):
        optimizer.zero_grad()
        train_epoch(model, mse_loss, features, targets)
        optimizer.step()

        pred = predict(model, features)
        loss = calc_loss(mse_loss, pred, targets).item()
        acc = calc_accuracy(pred, targets)
        epochs.append(epoch+1)
        losses.append(loss)
        accuracies.append(acc)

    rt = result_table(features, pred, targets)
    return Result(hyper=params, model=model, epochs=epochs, losses=losses, accuracies=accuracies, result_table=rt)


# %%
def run_torch(params: Hyper):
    print(f"Running Pytorch scenario: seed={params.seed}, optimizer={params.optimizer}, lr={params.lr}, nepochs={params.nepochs}")
    np.random.seed(params.seed)
    model = TwoLayerTorch(rand)
    optimizer = pytorch_optimizers[params.optimizer](model.parameters(), lr=params.lr)
    loss_fn = nn.MSELoss()
    features_ = torch.from_numpy(features)
    targets_ = torch.from_numpy(targets)

    epochs = []
    losses = []
    accuracies = []

    pred = model.forward(features_)
    loss = loss_fn(pred, targets_).item()
    acc = ((to_logic(pred) == to_logic(targets_)).sum() / pred.shape[0]).item()
    epochs.append(0)
    losses.append(loss)
    accuracies.append(acc)

    for epoch in tqdm(range(params.nepochs)):
        optimizer.zero_grad()
        pred = model.forward(features_)
        loss = loss_fn(pred, targets_)
        loss.backward()
        optimizer.step()

        pred = model.forward(features_)
        loss = loss_fn(pred, targets_).item()
        acc = ((to_logic(pred) == to_logic(targets_)).sum() / pred.shape[0]).item()
        epochs.append(epoch+1)
        losses.append(loss)
        accuracies.append(acc)

    rt = result_table_torch(features_, pred, targets_)
    return Result(hyper=params, model=model, epochs=epochs, losses=losses, accuracies=accuracies, result_table=rt)


# %% [markdown]
# ## SGD optimizer

# %%
h = Hyper(name='SGD', seed=12345, optimizer='SGD', lr=0.05, nepochs=200)

# %%
result1 = run_ag(h)
result2 = run_torch(h)

# %%
result1.result_table

# %%
result2.result_table

# %%
fig, axes = plt.subplots()
axes.plot(result1.epochs, result1.losses, label='AG')
axes.plot(result2.epochs, result2.losses, label='Torch')
axes.set_xscale("log")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.set_title("Loss vs Epoch")
axes.legend()
plt.show()

# %% [markdown]
# ## Adam optimizer

# %%
h = Hyper(name='Adam', seed=12345, optimizer='Adam', lr=0.05, nepochs=200)

# %%
result1 = run_ag(h)
result2 = run_torch(h)

# %%
result1.result_table

# %%
result2.result_table

# %%
fig, axes = plt.subplots()
axes.plot(result1.epochs, result1.losses, label='AG')
axes.plot(result2.epochs, result2.losses, label='Torch')
axes.set_xscale("log")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.set_title("Loss vs Epoch")
axes.legend()
plt.show()

# %% [markdown]
# ## AdamW optimizer

# %%
h = Hyper(name='AdamW', seed=12345, optimizer='AdamW', lr=0.05, nepochs=200)

# %%
result1 = run_ag(h)
result2 = run_torch(h)

# %%
result1.result_table

# %%
result2.result_table

# %%
fig, axes = plt.subplots()
axes.plot(result1.epochs, result1.losses, label='AG')
axes.plot(result2.epochs, result2.losses, label='Torch')
axes.set_xscale("log")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.set_title("Loss vs Epoch")
axes.legend()
plt.show()
