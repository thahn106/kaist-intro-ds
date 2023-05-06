# %% [markdown]
# ### Use this notebook for the given task of the term project.
#
# ### **_Make sure to delete all text cells before submission to avoid an unncessary increase of Turnitin similarity. That is, leave only the code cells._**
#

# %% [markdown]
# We will use only **'train.csv'**. Load the 'train.csv' file into a Pands dataframe.
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
train_data_raw.tail(1000)



# %% [markdown]
# Keep only the first two digits in the 'ClassificationID' attribute. For example, `9619001090` $\rightarrow$ `96`. If 'ClassificationID' has only nine (not ten) digits, you need to extract the first one digit (e.g., `505100000` $\rightarrow$ `5`).
#

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

# %% [markdown]
# Please note that the 'DeclarationOfficeID', 'PaymentType', 'BorderTransportMeans', and 'ExportationCountry' attributes are categorical, even though some of them have numerical forms. Then, let's factorize these attributes. Here, for each unique value in an attribute, assign consecutive integer values in the increasing order of the original values. Let's call these newly assigned integers as _codes_. As an example with 'ExportationCountry', `AE` $\rightarrow$ `0`.
#
# **Hint**: use `pd.factorize(train_df[...], sort=True)` for each of the attributes.
#
# ####Q1. What is the maximum code of the 'ExportationCountry' attribute?
#
# ####Q2. What is the maximum code of the 'DeclarationOfficeID' attribute?
#

# %%
pd.factorize(train_data["ClassificationID"], sort=True)[0]


# %% [markdown]
# Convert the 'ClassificationID' attribute (categorical) to a set of asymmetric **binary** variables. Use `pd.get_dummies(...)`, and we do **not** want to introduce redundancy in this process (i.e., be caureful with the `drop_first` option). Keep the original dataframe and store this new one as another dataframe.
#
# #### Q3. What is the total number of columns of this **new** dataframe?
#

# %%


# %% [markdown]
# Split the dataframe (immediately after **Q2**, i.e., before executing `get_dummies`) into `train_part` that contains the rows from January through September and `test_part` that contains the rows from October through December.
#
# Then, let's use `train_part` as the training set and `test_part` as the test set.
#
# ####Q4. How many examples (rows) are contained in `train_part`?
#

# %%


# %% [markdown]
# Let's consider a dumb classifier that **always** returns **0**.
#
# #### Q5. What is the accuracy of the dumb classifier on the test set `test_part`? Round the number to three decimal places (e.g., 0.67861352 $\rightarrow$ 0.679).
#

# %%


# %% [markdown]
# Let's implement a **k-nearest neighbor** classifier. To ease your implementation, please use `KNeighborsClassifier` of the scikit-learn library. Refer to the scikit-learn [manual](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). Use the default values for all parameters except `n_neighborsint` and `weights`.
#
# Note that we want to weight points by the inverse of their distance.
#
# #### Q6. Increase the number ($k$) of neighbors from **1** through **9**. At which value of $k$, is the accuracy of the classifier on the test set `test_part` maximized?
#
# #### Q7. What is the accuracy achieved in **Q6**? Round the number to three decimal places (e.g., 0.67861352 $\rightarrow$ 0.679).
#

# %%


# %% [markdown]
# Repeat the training and test procedures using the other dataframe obtained in **Q3**.
#
# #### Q8. Comparing the accuracy at the same value of $k$ as **Q6**~**Q7**, is the accuracy improved by using the other dataframe in **Q3**?
#

# %%
