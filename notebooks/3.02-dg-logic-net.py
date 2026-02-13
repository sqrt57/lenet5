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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lenet5.modeling.logic_net as net

# %%
x = np.linspace(-2, 3, 100)
fig, ax = plt.subplots()
ax.plot(x, net.activation(x), label="act")
ax.plot(x, net.activation_derivative(x), label="d act/dx")
ax.grid()
ax.legend()
plt.show()

# %%
targets = net.targets_xor
train_schedule = [
    (1000, 0.05),
    (100, .5),
    (10000, 0.05),
]
seed = 12345

# %% [markdown]
# # One Layer

# %%
np.random.seed(seed)
model = net.OneLayerBatch()
print(model.weights)
print(model.bias)

epoch = 0
epochs = []
losses = []
corrects = []

pred = model.forward(net.features)
epochs.append(epoch)
losses.append(net.loss(pred, targets))
corrects.append(net.correct_percent(pred, targets))

print(net.result_table(model, net.features, targets))
loss = net.loss(model.forward(net.features), targets)
print(f"Loss: {loss:.6f}")
loss_derivative = net.loss_derivative(model.forward(net.features), targets)
print(loss_derivative)

for (n_epochs, lr) in train_schedule:
    for i in range(n_epochs):
        epoch += 1
        
        model.zero_grad()
        pred = model.forward(net.features)
        model.backward(net.loss_derivative(pred, targets))
        model.subtract_grad(lr)

        pred = model.forward(net.features)
        epochs.append(epoch)
        losses.append(net.loss(pred, targets))
        corrects.append(net.correct_percent(pred, targets))

print()
print(model.weights)
print(model.bias)
print(net.result_table(model, net.features, targets))
loss = net.loss(model.forward(net.features), targets)
print(f"Loss: {loss:.6f}")

# %%
fig, ax = plt.subplots()
ax.plot(epochs, losses)
ax1 = ax.twinx()
ax1.plot(epochs, corrects, color='red')
plt.show()

# %% [markdown]
# # Two Layers

# %%
np.random.seed(seed)
model = net.TwoLayerBatch()
# model.weights1 = np.array([[1., -1.], [-1., 1.]])
# model.bias1 = np.array([0., 0.])

print(model.weights1)
print(model.bias1)
print(model.weights2)
print(model.bias2)

epoch = 0
epochs = []
losses = []
corrects = []
weights1 = []
biases1 = []
weights2 = []
biases2 = []
weights1_gradient = []
biases1_gradient = []
weights2_gradient = []
biases2_gradient = []

pred = model.forward(net.features)
epochs.append(epoch)
losses.append(net.loss(pred, targets))
corrects.append(net.correct_percent(pred, targets))

weights1.append(model.weights1.copy())
biases1.append(model.bias1.copy())
weights2.append(model.weights2.copy())
biases2.append(model.bias2.copy())
weights1_gradient.append(model.weights1_grad.copy())
biases1_gradient.append(model.bias1_grad.copy())
weights2_gradient.append(model.weights2_grad.copy())
biases2_gradient.append(model.bias2_grad.copy())

print(net.result_table(model, net.features, targets))
loss = net.loss(model.forward(net.features), targets)
print(f"Loss: {loss:.6f}")
loss_derivative = net.loss_derivative(model.forward(net.features), targets)
print(loss_derivative)

for (n_epochs, lr) in train_schedule:
    for i in range(n_epochs):
        epoch += 1
        
        model.zero_grad()
        pred = model.forward(net.features)
        model.backward(net.loss_derivative(pred, targets))
        model.subtract_grad(lr)

        pred = model.forward(net.features)
        epochs.append(epoch)
        losses.append(net.loss(pred, targets))
        corrects.append(net.correct_percent(pred, targets))

        weights1.append(model.weights1.copy())
        biases1.append(model.bias1.copy())
        weights2.append(model.weights2.copy())
        biases2.append(model.bias2.copy())
        weights1_gradient.append(model.weights1_grad.copy())
        biases1_gradient.append(model.bias1_grad.copy())
        weights2_gradient.append(model.weights2_grad.copy())
        biases2_gradient.append(model.bias2_grad.copy())

print()
print(model.weights1)
print(model.bias1)
print(model.weights2)
print(model.bias2)
print(net.result_table(model, net.features, targets))
loss = net.loss(model.forward(net.features), targets)
print(f"Loss: {loss:.6f}")

# %%
print(model.weights1_grad)
print(model.bias1_grad)
print(model.weights2_grad)
print(model.bias2_grad)

# %%
net.loss_derivative(model.forward(net.features), targets)

# %%
fig, ax = plt.subplots()
ax1 = ax.twinx()
ax1.plot(epochs, corrects, color='red')
ax.plot(epochs, losses)
plt.show()

# %%
fig, axs = plt.subplots(2,2)
fig.set_size_inches(12,10)

axs[0,0].plot(epochs, [w[0,0] for w in weights1], label='w00')
axs[0,0].plot(epochs, [w[0,1] for w in weights1], label='w01')
axs[0,0].plot(epochs, [w[1,0] for w in weights1], label='w10')
axs[0,0].plot(epochs, [w[1,1] for w in weights1], label='w11')
axs[0,0].plot(epochs, [w[0,0] for w in weights1_gradient], label='w00g')
axs[0,0].plot(epochs, [w[0,1] for w in weights1_gradient], label='w01g')
axs[0,0].plot(epochs, [w[1,0] for w in weights1_gradient], label='w10g')
axs[0,0].plot(epochs, [w[1,1] for w in weights1_gradient], label='w11g')
axs[0,0].legend()

axs[1,0].plot(epochs, [w[0,0] for w in weights2], label='w0')
axs[1,0].plot(epochs, [w[1,0] for w in weights2], label='w1')
axs[1,0].plot(epochs, [w[0,0] for w in weights2_gradient], label='w0g')
axs[1,0].plot(epochs, [w[1,0] for w in weights2_gradient], label='w1g')
axs[1,0].legend()

axs[0,1].plot(epochs, [b[0] for b in biases1], label='b0')
axs[0,1].plot(epochs, [b[1] for b in biases1], label='b1')
axs[0,1].plot(epochs, [b[0] for b in biases1_gradient], label='b0g')
axs[0,1].plot(epochs, [b[1] for b in biases1_gradient], label='b1g')
axs[0,1].legend()

axs[1,1].plot(epochs, [b[0] for b in biases2], label='b')
axs[1,1].plot(epochs, [b[0] for b in biases2_gradient], label='bg')
axs[1,1].legend()

fig.tight_layout()
plt.show()

# %%

# %%
