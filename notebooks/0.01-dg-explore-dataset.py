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
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
# %%
np.random.seed(391432719)

# %%
dataset = []
for path in Path('../data/raw').glob('*.bmp'):
    if not path.is_file(): continue
    name = path.stem
    digit = int(name[0])
    im = Image.open(path)
    dataset.append((digit, name, im))
len(dataset)

# %%
fig, axs = plt.subplots(10,12)
fig.set_size_inches(12,10)
for i in range(120):
    img = dataset[i][2]
    label = dataset[i][0]
    ax = axs.flat[i]
    ax.imshow(img, cmap='gray', interpolation='none')
    ax.set_yticks([], [])
    ax.set_xticks([], [])
    ax.set_xlabel(str(label))
fig.tight_layout()
plt.show()

# %%
dataset[5][2].__dict__
