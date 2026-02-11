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
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import lenet5.dataset as ds

# %%
np.random.seed(600789589)

# %%
dataset = ds.load_raw_data(Path('../data/raw'))

# %%
print(dataset.features.shape)
print(dataset.features.dtype)
print(np.unique(dataset.features, return_counts=True))

# %%
print(dataset.labels.shape)
print(dataset.labels.dtype)
print(np.unique(dataset.labels, return_counts=True))

# %%
print(dataset.metadata.shape)
print(dataset.metadata.dtype)
print(dataset.metadata[0])

# %%
Path('../data/processed').mkdir(parents=True, exist_ok=True)
ds.save_dataset(Path('../data/processed/source.npz'), dataset)
