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
column_handler = {
    "ID": "pass",
    "IssueDateTime": "datetime",
    "DeclarationOfficeID": "dummy",
    "ProcessType": "dummy",
    "TransactionNature": "dummy",
    "Type": "dummy",
    "PaymentType": "dummy",
    "BorderTransportMeans": "dummy",
    "DeclarerID": "drop_an",
    "ImporterID": "drop_an",
    "SellerID": "drop_an",
    "ExpressID": "drop_an",
    "ClassificationID": "dummy",
    "ExportationCountry": "dummy",
    "OriginCountry": "dummy",
    "TaxRate": "numeric",
    "DutyRegime": "dummy",
    "DisplayIndicator": "dummy",
    "TotalGrossMassMeasure(KG)": "numeric",
    "AdValoremTaxBaseAmount(Won)": "numeric",
    "Fake": "pass",
    "data_type": "pass",
}


def preprocess_data(train_data_raw, test_data_raw):
    train_data = train_data_raw.copy()
    test_data = test_data_raw.copy()
    train_data["data_type"] = "train"
    test_data["data_type"] = "test"
    test_data["Fake"] = pd.NA
    data = pd.concat([train_data, test_data], axis=0)

    for col in data.columns.values:
        if col not in column_handler:
            print(f"Column {col} not in column_handler")
            data = data.drop([col], axis=1)
        elif column_handler[col] == "drop" or column_handler[col] == "drop_an":
            data = data.drop([col], axis=1)
        elif column_handler[col] == "dummy":
            data[col] = pd.factorize(data[col], sort=True)[0]
            data = pd.get_dummies(data, prefix=[col], columns=[col])
        elif column_handler[col] == "datetime":
            data[col] = data[col].dt.month / 12
        elif column_handler[col] == "numeric":
            min_val = data[col].min()
            max_val = data[col].max()
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            pass

    train_data = data.loc[data["data_type"] == "train"]
    test_data = data.loc[data["data_type"] == "test"]

    train_data = train_data.drop(["data_type"], axis=1)
    test_data = test_data.drop(["data_type"], axis=1)

    ID = test_data["ID"]
    train_data = train_data.drop(["ID"], axis=1)
    test_data = test_data.drop(["ID"], axis=1)

    train_labels = train_data["Fake"]
    train_data = train_data.drop(["Fake"], axis=1)
    test_data = test_data.drop(["Fake"], axis=1)

    return train_data, train_labels, test_data, ID


train_data, train_labels, test_data, ID = preprocess_data(train_data_raw, test_data_raw)

train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)

# %%
model_1 = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(
            200, activation="relu", input_shape=(train_data.shape[1],)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model_1.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# %%
model_1.fit(train_data, train_labels, epochs=10)

# %%
predictions = model_1.predict(test_data)

result_df = pd.DataFrame({"ID": ID, "Fake": predictions.flatten()})
result_df["Fake"] = result_df["Fake"].apply(lambda x: 1 if x > 0.5 else 0)
result_df.to_csv("20160840_submission.csv", index=False)
