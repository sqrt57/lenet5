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

# %%
training = dataset.load_train_dataset().torch32()
test = dataset.load_test_dataset().torch32()

# %%
Hyper = collections.namedtuple('Hyper', 'name seed model optimizer lr optimizer_kwargs nepochs')
Result = collections.namedtuple('Result', 'hyper model epochs train_losses test_losses train_accuracies test_accuracies')


# %%
def activation(x):
    return torch.tanh(2/3 * x) / math.tanh(2/3)

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        m = 2.4/257
        self.w = nn.Parameter((torch.rand(16*16,10)*2-1)*m)
        self.b = nn.Parameter((torch.rand(10)*2-1)*m)
    
    def forward(self, x):
        d = x.shape[0]
        assert x.shape == (d, 1, 16, 16)
        
        x_flat = x.view(d, 1, 256)
        result = activation(x_flat @ self.w + self.b.view(1, 1, 10))
        assert result.shape == (d, 1, 10)
        
        return torch.flatten(result, 1)


# %%
loss_fn = nn.CrossEntropyLoss()


# %%
def run_scenario(hyper: Hyper, loss_fn, training, test):
    print(f"Running Pytorch scenario {hyper.name}: model={hyper.model.__name__} seed={hyper.seed} lr={hyper.lr} nepochs={hyper.nepochs}")
    torch.random.manual_seed(hyper.seed)
    model = hyper.model()
    optimizer = hyper.optimizer(model.parameters(), lr=hyper.lr, **hyper.optimizer_kwargs)

    epochs = []
    losses = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    def batch(features_batch, labels_batch):
        optimizer.zero_grad()
        pred = model.forward(features_batch)
        loss = loss_fn(pred, labels_batch)
        loss.backward()
        optimizer.step()

    def log_metrics(epoch):
        epochs.append(epoch)

        pred = model.forward(training.features)
        train_losses.append(loss_fn(pred, training.labels).item())
        train_accuracies.append((pred.argmax(1)==training.labels).sum().item() * 100 / pred.shape[0])

        pred = model.forward(test.features)
        test_losses.append(loss_fn(pred, test.labels).item())
        test_accuracies.append((pred.argmax(1)==test.labels).sum().item() * 100 / pred.shape[0])

    log_metrics(0)

    for epoch in tqdm(range(hyper.nepochs)):
        for i in range(training.features.shape[0]):
            batch(training.features[i:i+1,:,:,:], training.labels[i:i+1])

        log_metrics(epoch+1)

    return Result(hyper=hyper, model=model, epochs=epochs,
                  train_losses=train_losses, test_losses=test_losses,
                  train_accuracies=train_accuracies, test_accuracies=test_accuracies)


# %%
seed = 93862057
h1 = Hyper(name='Net1', seed=seed, model=Net1, optimizer=torch.optim.Adam, lr=1e-3, optimizer_kwargs={'betas':(0., 0.999)}, nepochs=30)

# %%
result = run_scenario(h1, loss_fn, training, test)

# %%
#result

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,  4))

ax1.plot(result.epochs, result.train_accuracies, label='train')
ax1.plot(result.epochs, result.test_accuracies, label='test')
ax1.set_ylim(60., 100.)
ax1.set_xlabel("Epoch")
ax1.set_title("Accuracy")
ax1.grid(True)
ax1.legend()

ax2.plot(result.epochs, result.train_losses, label='train')
ax2.plot(result.epochs, result.test_losses, label='test')
# ax2.set_ylim(60., 100.)
ax2.set_xlabel("Epoch")
ax2.set_title("Loss")
ax2.grid(True)
ax2.legend()

plt.plot()


# %%
def find_errors(model, dataset):
    pred = model.forward(dataset.features)
    return (pred.argmax(1) != dataset.labels).nonzero()[:,0]


# %%
find_errors(result.model, training)

# %%
find_errors(result.model, test).shape

# %%
dataset = training
errors = find_errors(result.model, dataset)
fig, axs = plt.subplots(1,2)

for i in range(errors.shape[0]):
    ax = axs.flat[i]
    img = dataset.features[errors[i]].squeeze()
    img = img - img.min()
    img = img / img.max() * 255
    img = img.int()
    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.set_xlabel(str(dataset.labels[errors[i]].item()))
    
for i in range(errors.shape[0], 2):
    ax = axs.flat[i]
    ax.imshow(torch.ones((16,16)).byte()*200, cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    
plt.tight_layout()
plt.subplots_adjust(wspace=None, hspace=None)
plt.show()

# %%
dataset = test
errors = find_errors(result.model, dataset)
fig, axs = plt.subplots(5, 10, figsize=(10,  4))

for i in range(errors.shape[0]):
    ax = axs.flat[i]
    img = dataset.features[errors[i]].squeeze()
    img = img - img.min()
    img = img / img.max() * 255
    img = img.int()
    ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.set_xlabel(str(dataset.labels[errors[i]].item()))

for i in range(errors.shape[0], 50):
    ax = axs.flat[i]
    ax.imshow(torch.ones((16,16)).byte()*200, cmap="gray", vmin=0, vmax=255)
    ax.set_yticks([], [])
    ax.set_xticks([], [])

plt.tight_layout()
plt.subplots_adjust(wspace=None, hspace=None)
plt.show()
