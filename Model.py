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

# import data
train_file = os.path.join(os.getcwd(), "data", "train.csv")
# test_file = os.path.join(os.getcwd(), "data", "test.csv")

train_data_raw = pd.read_csv(train_file, parse_dates=["IssueDateTime"])
# test_data_raw = pd.read_csv(test_file, parse_dates=["IssueDateTime"])

# %%
# Verify Tensorflow is installed correctly
# https://chancoding.tistory.com/223
from tensorflow import keras
from tensorflow.python.client import device_lib

print(f"{tf.__version__=}, {keras.__version__=}")

# Verify GPU is available
device_lib.list_local_devices()


# %%
# Preprocessing

def preprocess(data_raw):
    data = data_raw.copy()
    
    return data



# %%
