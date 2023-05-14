# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model
#
# Written as a synced .py file, executable as a notebook by the Jupytext extension.

# %%
# Basic imports and file paths
import os

import numpy as np
import pandas as pd
import tensorflow as tf

# Verify Tensorflow is installed correctly
from tensorflow import keras
from tensorflow.python.client import device_lib

print(f"{tf.__version__=}, {keras.__version__=}")

# Verify GPU is available
print(device_lib.list_local_devices())

# import data
train_file = os.path.join(os.getcwd(), "data", "train.csv")
test_file = os.path.join(os.getcwd(), "data", "test.csv")

train_data_raw = pd.read_csv(train_file, parse_dates=["IssueDateTime"])
test_data_raw = pd.read_csv(test_file, parse_dates=["IssueDateTime"])

# %%
# Preprocessing


def preprocess(data_raw):
    data = data_raw.copy()
    return data
