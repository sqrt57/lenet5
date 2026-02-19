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

# %% [markdown]
# ## Imports

# %%
import sys
sys.path.append("..")

# %%
import numpy as np
import matplotlib.pyplot as plt

import lenet5.dataset as ds

# %% [markdown]
# ## Run data preprocessing

# %%
processed = ds.main()
source_dataset = processed["source_dataset"]
augmented_dataset = processed["augmented_dataset"]
train_dataset = processed["train_dataset"]
test_dataset = processed["test_dataset"]

# %% [markdown]
# ## Source dataset, 120 images sized 16x13, 12 images for each of 10 digit classes

# %%
print(source_dataset.features.shape)
print(source_dataset.features.dtype)
print(np.unique(source_dataset.features, return_counts=True))

# %%
print(source_dataset.labels.shape)
print(source_dataset.labels.dtype)
print(np.unique(source_dataset.labels, return_counts=True))

# %%
print(type(source_dataset.metadata))
print(len(source_dataset.metadata))
print(source_dataset.metadata[5])

# %%
fig, axs = plt.subplots(10,12)
fig.set_size_inches(12,10)
for i in range(120):
    img = source_dataset.features[i]
    label = source_dataset.labels[i]
    ax = axs.flat[i]
    ax.imshow(img, cmap='gray', interpolation='none')
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.set_xlabel(str(label))
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Train dataset, 320 images sized 16x16, 32 images for each of 10 digit classes

# %%
print(train_dataset.features.shape)
print(train_dataset.features.dtype)
print(np.unique(train_dataset.features, return_counts=True))

# %%
print(train_dataset.labels.shape)
print(train_dataset.labels.dtype)
print(np.unique(train_dataset.labels, return_counts=True))

# %%
print(type(train_dataset.metadata))
print(len(train_dataset.metadata))
print(train_dataset.metadata[5])

# %%
fig, axs = plt.subplots(32,10)
fig.set_size_inches(10,32)
for i in range(320):
    img = train_dataset.features[i]
    label = train_dataset.labels[i]
    ax = axs.flat[i]
    ax.imshow(img, cmap='gray', interpolation='none')
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.set_xlabel(str(label))
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Test dataset, 160 images sized 16x16, 16 images for each of 10 digit classes

# %%
print(test_dataset.features.shape)
print(test_dataset.features.dtype)
print(np.unique(test_dataset.features, return_counts=True))

# %%
print(test_dataset.labels.shape)
print(test_dataset.labels.dtype)
print(np.unique(test_dataset.labels, return_counts=True))

# %%
print(type(test_dataset.metadata))
print(len(test_dataset.metadata))
print(test_dataset.metadata[5])

# %%
fig, axs = plt.subplots(16,10)
fig.set_size_inches(10,16)
for i in range(160):
    img = test_dataset.features[i]
    label = test_dataset.labels[i]
    ax = axs.flat[i]
    ax.imshow(img, cmap='gray', interpolation='none')
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.set_xlabel(str(label))
fig.tight_layout()
plt.show()  
