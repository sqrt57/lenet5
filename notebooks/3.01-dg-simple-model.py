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
import matplotlib.pyplot as plt

import lenet5.dataset as ds

# %%
train = ds.load_train_dataset()
train_features = train.features.reshape((320, 16*16))
train_targets = np.eye(10)[train.labels]

test = ds.load_test_dataset()
test_features = test.features.reshape((160, 16*16))
test_targets = np.eye(10)[test.labels]

# %% [markdown]
# # Linear regression

# %%
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model

linear = Pipeline([
    ('variance_threshold', VarianceThreshold(threshold=(.9 * (1 - .9)))), 
    ('regression', linear_model.LinearRegression()),
])
linear.fit(train_features, train_targets)

# %%
test_preds = linear.predict(test_features)
test_error = ((test_preds - test_targets)**2).sum() / len(test_targets)
test_acc = test_preds.argmax(axis=1) == test.labels
print(f"Test error: {test_error:.4f}, Test accuracy: {test_acc.mean() * 100:.2f}")

# %% [markdown]
# # Support vector machine

# %%
from sklearn import svm

svm_model = svm.SVC()
svm_model.fit(train_features, train.labels)
# %%
train_preds = svm_model.predict(train_features)
train_error = (train_preds != train.labels).sum() * 2 / len(train_targets)
train_acc = train_preds == train.labels
print(f"Train error: {train_error:.4f}, Train accuracy: {train_acc.mean() * 100:.2f}")

# %%
test_preds = svm_model.predict(test_features)
test_error = (test_preds != test.labels).sum()*2 / len(test_targets)
test_acc = test_preds == test.labels
print(f"Test error: {test_error:.4f}, Test accuracy: {test_acc.mean() * 100:.2f}")

# %% [markdown]
# # Lasso

# %%
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model

clf = Pipeline([
    ('variance_threshold', VarianceThreshold(threshold=(.9 * (1 - .9)))), 
    ('regression', linear_model.Lasso(alpha=1.0)),
])
# clf = linear_model.Lasso(alpha=1.0)
clf.fit(train_features, train_targets)

# %%
train_preds = clf.predict(train_features)
train_error = ((train_preds - train_targets)**2).sum() / len(train_targets)
train_acc = train_preds.argmax(axis=1) == train.labels
print(f"Train error: {train_error:.4f}, Train accuracy: {train_acc.mean() * 100:.2f}")

# %%
test_preds = clf.predict(test_features)
test_error = ((test_preds - test_targets)**2).sum() / len(test_targets)
test_acc = test_preds.argmax(axis=1) == test.labels
print(f"Test error: {test_error:.4f}, Test accuracy: {test_acc.mean() * 100:.2f}")

# %% [markdown]
# # Ridge

# %%
from sklearn.kernel_ridge import KernelRidge
krr = KernelRidge(alpha=1.0)
krr.fit(train_features, train_targets)

# %%
train_preds = krr.predict(train_features)
train_error = ((train_preds - train_targets)**2).sum() / len(train_targets)
train_acc = train_preds.argmax(axis=1) == train.labels
print(f"Train error: {train_error:.4f}, Train accuracy: {train_acc.mean() * 100:.2f}")

# %%
test_preds = krr.predict(test_features)
test_error = ((test_preds - test_targets)**2).sum() / len(test_targets)
test_acc = test_preds.argmax(axis=1) == test.labels
print(f"Test error: {test_error:.4f}, Test accuracy: {test_acc.mean() * 100:.2f}")

# %% [markdown]
# # K-neghbours

# %%
from sklearn.neighbors import KNeighborsClassifier

nbrs = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
nbrs.fit(train_features, train.labels)
# %%
train_preds = nbrs.predict(train_features)
train_error = (train_preds != train.labels).sum() * 2 / len(train_targets)
train_acc = train_preds == train.labels
print(f"Train error: {train_error:.4f}, Train accuracy: {train_acc.mean() * 100:.2f}")

# %%
test_preds = nbrs.predict(test_features)
test_error = (test_preds != test.labels).sum() * 2 / len(test_targets)
test_acc = test_preds == test.labels
print(f"Test error: {test_error:.4f}, Test accuracy: {test_acc.mean() * 100:.2f}")
