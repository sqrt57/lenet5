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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import lenet5.dataset as dataset
import lenet5.modeling.lenet as models
from lenet5.modeling.train import Hyper, Result, Trainer

# %%
training = dataset.load_train_dataset().torch32()
test = dataset.load_test_dataset().torch32()

# %%
i = 15
n = 3
training_one = dataset.DataSet(training.features[i:i+n], training.labels[i:i+n], training.metadata[i:i+n])

# %%
trainer = Trainer(training, test)

# %%
seed = 682200895
# seeds = [
#     85789640,
#     898739203,
#     661922000,
#     161962468,
#     755185520,
#     424894101,
#     625219363,
#     94554931,
#     822250535,
#     52246560,
# ]
optimizer = torch.optim.Adam
lr = 5e-4
optimizer_kwargs = { 'betas': (0.9, 0.999) }
nepochs = 2000
batch_size = 32
random_order = True
hypers = [
    Hyper(name=f'Net1', seed=seed, model=models.Net1, optimizer=optimizer, lr=lr, optimizer_kwargs=optimizer_kwargs, nepochs=nepochs, batch_size=batch_size, random_order=random_order),
    Hyper(name=f'Net2', seed=seed, model=models.Net2, optimizer=optimizer, lr=lr, optimizer_kwargs=optimizer_kwargs, nepochs=nepochs, batch_size=batch_size, random_order=random_order),
    Hyper(name=f'Net3', seed=seed, model=models.Net3, optimizer=optimizer, lr=lr, optimizer_kwargs=optimizer_kwargs, nepochs=nepochs, batch_size=batch_size, random_order=random_order),
    Hyper(name=f'Net4', seed=seed, model=models.Net4, optimizer=optimizer, lr=lr, optimizer_kwargs=optimizer_kwargs, nepochs=nepochs, batch_size=batch_size, random_order=random_order),
    Hyper(name=f'Net5', seed=seed, model=models.Net5, optimizer=optimizer, lr=lr, optimizer_kwargs=optimizer_kwargs, nepochs=nepochs, batch_size=batch_size, random_order=random_order),
]

# %%
results = []
for hyper in hypers:
    results.append(trainer.run_scenario(hyper))

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,  8))

for result in results:
    ax1.plot(result.epochs, result.train_accuracies, label=result.hyper.name)
ax1.set_ylim(60., 100.)
# ax1.set_ylim(0., 100.)
ax1.set_xlabel("Epoch")
ax1.set_title("Train accuracy")
ax1.grid(True)
ax1.legend()

for result in results:
    ax2.plot(result.epochs, result.test_accuracies, label=result.hyper.name)
ax2.set_ylim(60., 100.)
# ax2.set_ylim(0., 100.)
ax2.set_xlabel("Epoch")
ax2.set_title("Test accuracy")
ax2.grid(True)
ax2.legend()

for result in results:
    ax3.plot(result.epochs, result.train_losses, label=result.hyper.name)
ax3.set_ylim(0., 5.)
ax3.set_xlabel("Epoch")
ax3.set_title("Train loss")
ax3.grid(True)
ax3.legend()

for result in results:
    ax4.plot(result.epochs, result.test_losses, label=result.hyper.name)
ax4.set_ylim(0., 5.)
ax4.set_xlabel("Epoch")
ax4.set_title("Test loss")
ax4.grid(True)
ax4.legend()

fig.tight_layout()
plt.plot()


# %%
def find_errors(model, ds):
    pred = model.forward(ds.features * 2 - 1)
    wrong = (pred.argmax(1) != ds.labels).nonzero()[:,0]
    return (wrong, pred[wrong].argmax(1))


# %%
sum(p.numel() for p in results[2].model.parameters() if p.requires_grad)

# %%
result = results[4]

# %%
find_errors(result.model, training)

# %%
find_errors(result.model, test)[0].shape

# %%
current_dataset = training
(errors, wrong_pred) = find_errors(result.model, current_dataset)
fig, axs = plt.subplots(2,10, figsize=(10,2))

for i in range(min(errors.shape[0], 20)):
    ax = axs.flat[i]
    img = current_dataset.features[errors[i]].squeeze()
    img = img - img.min()
    img = img / img.max() * 255
    img = img.int()
    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    target_label = current_dataset.labels[errors[i]].item()
    pred_label = wrong_pred[i].item()
    ax.set_xlabel(f"{pred_label} ({target_label})")
    
for i in range(errors.shape[0], 20):
    ax = axs.flat[i]
    ax.imshow(torch.ones((16,16)).byte()*200, cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    
plt.tight_layout()
plt.subplots_adjust(wspace=None, hspace=None)
plt.show()

# %%
current_dataset = test
(errors, wrong_pred) = find_errors(result.model, current_dataset)
fig, axs = plt.subplots(6, 10, figsize=(10,  5))

for i in range(min(errors.shape[0], 60)):
    ax = axs.flat[i]
    img = current_dataset.features[errors[i]].squeeze()
    img = img - img.min()
    img = img / img.max() * 255
    img = img.int()
    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    target_label = current_dataset.labels[errors[i]].item()
    pred_label = wrong_pred[i].item()
    ax.set_xlabel(f"{pred_label} ({target_label})")

for i in range(errors.shape[0], 60):
    ax = axs.flat[i]
    ax.imshow(torch.ones((16,16)).byte()*200, cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([], [])
    ax.set_xticks([], [])

plt.tight_layout()
plt.subplots_adjust(wspace=None, hspace=None)
plt.show()

# %%
