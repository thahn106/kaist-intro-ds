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
# # Preprocessing
#

# %%
# Basic imports and file paths

import numpy as np
import pandas as pd
import os

# import data
train_file = os.path.join(os.getcwd(), "data", "train.csv")
test_file = os.path.join(os.getcwd(), "data", "test.csv")
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

