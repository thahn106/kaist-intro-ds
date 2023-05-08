# %% [markdown]
# # Model Building
#
# Written as a synced .py file, executable as a notebook by the Jupytext extension.
#

# %%
# Basic imports and file paths

import numpy as np
import pandas as pd
import os

# import data
train_file = os.path.join(os.getcwd(), "data", "train.csv")
# test_file = os.path.join(os.getcwd(), "data", "test.csv")

train_data_raw = pd.read_csv(train_file, parse_dates=["IssueDateTime"])
# test_data_raw = pd.read_csv(test_file, parse_dates=["IssueDateTime"])

# %%
train_data = train_data_raw.drop(
    [
        "ID",
        "ProcessType",
        "TransactionNature",
        "Type",
        "DeclarerID",
        "ImporterID",
        "SellerID",
        "ExpressID",
        "OriginCountry",
        "DisplayIndicator",
        "DutyRegime",
    ],
    axis=1,
)
train_data["IssueDateTime"] = train_data["IssueDateTime"].dt.month
train_data["ClassificationID"] = train_data["ClassificationID"].apply(
    lambda x: str(x)[:-8]
)  # drop last 8 characters
train_data.info()

# %%
categorical_columns = [
    "DeclarationOfficeID",
    "PaymentType",
    "BorderTransportMeans",
    "ExportationCountry",
]
for col in categorical_columns:
    train_data[col] = pd.factorize(train_data[col], sort=True)[0]

# Questions 1, 2
print(f"{train_data['ExportationCountry'].max()=}")
print(f"{train_data['DeclarationOfficeID'].max()=}")


# %%
# Question 3
train_data_dummy = pd.get_dummies(
    train_data, prefix=["ClassificationID"], columns=["ClassificationID"]
)
train_data_dummy.head()
print(f"{train_data_dummy.columns.size=}")


# %%
# Question 4
train_part = train_data[train_data["IssueDateTime"] < 10]
test_part = train_data[train_data["IssueDateTime"] >= 10]

print(f"{len(train_data)=}")
print(f"{len(train_part)=}")
print(f"{len(test_part)=}")


# %%
# Question 5
from sklearn.metrics import accuracy_score

dumb_pred = np.zeros(len(test_part))
dumb_true = test_part["Fake"]

print(f"{accuracy_score(dumb_true, dumb_pred):.3f}")


# %%
# Questions 6, 7
from sklearn.neighbors import KNeighborsClassifier

train_part_data = train_part.drop(["Fake"], axis=1)
train_part_labels = train_part["Fake"]
test_part_data = test_part.drop(["Fake"], axis=1)
knn_true = test_part["Fake"]

for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(train_part_data, train_part_labels)
    knn_pred = knn.predict(test_part_data)
    print(f"Accuracy for k={k}: {accuracy_score(knn_true, knn_pred):.3f}.")


# %%
# Question 8 (With dummies)
K_FIXED = 7
train_part_dummy = train_data_dummy[train_data_dummy["IssueDateTime"] < 10]
test_part_dummy = train_data_dummy[train_data_dummy["IssueDateTime"] >= 10]

knn_true = test_part_dummy["Fake"]
knn = KNeighborsClassifier(n_neighbors=K_FIXED, weights="distance")
knn.fit(train_part_dummy.drop(["Fake"], axis=1), train_part_dummy["Fake"])
knn_pred = knn.predict(test_part_dummy.drop(["Fake"], axis=1))
print(f"Accuracy for k={K_FIXED}: {accuracy_score(knn_true, knn_pred):.3f}.")
