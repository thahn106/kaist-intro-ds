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
# # Final Model Building
#
# Written as a synced .py file, executable as a notebook by the Jupytext extension.
#

# %%
# Basic imports and file paths

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict

# import data
train_file = os.path.join(os.getcwd(), "data", "train.csv")
test_file = os.path.join(os.getcwd(), "data", "test.csv")

train_data_raw = pd.read_csv(train_file, parse_dates=["IssueDateTime"])
test_data_raw = pd.read_csv(test_file, parse_dates=["IssueDateTime"])


# %%
# Baseline Experiment Class
class Experiment:
    def __init__(self, train_data_raw, test_data_raw):
        self.train_data_raw = train_data_raw
        self.test_data_raw = test_data_raw
        self.reset()

    def reset(self):
        self.train_data = self.train_data_raw.copy()
        self.test_data = self.test_data_raw.copy()
        self.newcol_count = 0

    def drop_columns(self, columns):
        self.train_data.drop(columns=columns, inplace=True)
        self.test_data.drop(columns=columns, inplace=True)

    def factorize_columns(self, columns):
        for column in columns:
            self.train_data[column] = pd.factorize(self.train_data[column])[0]
            self.test_data[column] = pd.factorize(self.test_data[column])[0]


# %%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def evaluate_features(self, k=10):
    train_part_data = self.train_data.drop(["Fake"], axis=1)
    train_part_labels = self.train_data["Fake"]

    bestfeatures = SelectKBest(score_func=chi2, k=k)
    fit = bestfeatures.fit(train_part_data, train_part_labels)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(train_part_data.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["Specs", "Score"]  # naming the dataframe columns
    print(featureScores.nlargest(k, "Score"))


Experiment.evaluate_features = evaluate_features

# %%
##
ADJ_MODE_BASERATE = 0
ADJ_MODE_EQUAL = 1

DN = {
    ADJ_MODE_BASERATE: 4,
    ADJ_MODE_EQUAL: 1,
}
DP = {
    ADJ_MODE_BASERATE: 1,
    ADJ_MODE_EQUAL: 1,
}

BASE_RATE = 0.21


def rate_condense(self, columns, adj_amount=1, adj_mode=ADJ_MODE_BASERATE):
    rate_dict = defaultdict(lambda: [0, 0])
    for index, row in self.train_data.iterrows():
        key = tuple(row[columns])
        if row["Fake"]:
            rate_dict[key][1] += 1
        else:
            rate_dict[key][0] += 1

    rates = defaultdict(lambda: BASE_RATE)
    dn = DN[adj_mode] * adj_amount
    dp = DP[adj_mode] * adj_amount
    for key in rate_dict:
        neg, pos = rate_dict[key]
        neg += dn
        pos += dp
        rates[key] = pos / (neg + pos)

    newcol_name = "rate" + str(self.newcol_count)
    self.newcol_count += 1

    self.train_data[newcol_name] = self.train_data[columns].apply(
        lambda x: rates[tuple(x)], axis=1
    )
    self.train_data.drop(columns=columns, inplace=True)

    self.test_data[newcol_name] = self.test_data[columns].apply(
        lambda x: rates[tuple(x)], axis=1
    )
    self.test_data.drop(columns=columns, inplace=True)


Experiment.rate_condense = rate_condense

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras


def test_suite(self, tf_flag=False, verbose=False):
    if verbose:
        print("Testing...")

    train_data = self.train_data
    test_data = self.test_data

    train_part, test_part = train_test_split(train_data, test_size=0.2, random_state=42)

    train_part_data = train_part.drop(["Fake"], axis=1)
    train_part_labels = train_part["Fake"]
    test_part_data = test_part.drop(["Fake"], axis=1)
    test_part_labels = test_part["Fake"]

    max_score = 0

    for k in range(1, 12):
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(train_part_data, train_part_labels)
        knn_pred = knn.predict(test_part_data)
        score = accuracy_score(test_part_labels, knn_pred)
        max_score = max(max_score, score)
        if verbose:
            print(f"Accuracy for k={k}: {score:.3f}.")

    for k in range(4, 9):
        estimators = 2**k
        rf = RandomForestClassifier(n_estimators=estimators)
        rf.fit(train_part_data, train_part_labels)
        rf_pred = rf.predict(test_part_data)
        score = accuracy_score(test_part_labels, rf_pred)
        max_score = max(max_score, score)
        if verbose:
            print(f"Accuracy for n_estimators={k}: {score:.3f}.")

    if tf_flag:
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    64, activation="relu", input_shape=[len(train_part_data.keys())]
                ),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        model.fit(
            train_part_data,
            train_part_labels,
            epochs=10,
            batch_size=32,
            verbose=verbose,
        )

        test_loss, test_acc = model.evaluate(
            test_part_data, test_part_labels, verbose=verbose
        )

        max_score = max(max_score, test_acc)
        if verbose:
            print(f"Accuracy for tf: {test_acc:.3f}.")

    if verbose:
        print(f"Max score: {max_score:.3f}.")

    return max_score


Experiment.test_suite = test_suite

# %%
es = Experiment(train_data_raw, test_data_raw)
es_drop_columns = [
    "ID",
    "ProcessType",
    "Type",
    "DeclarerID",
    "ImporterID",
    "SellerID",
    "ExpressID",
    "DisplayIndicator",
    "IssueDateTime",
    "ClassificationID",
    # "AdValoremTaxBaseAmount(Won)",
    # "TotalGrossMassMeasure(KG)",
]
es_categorical_columns = [
    "TransactionNature",
    # "DeclarationOfficeID",
    "PaymentType",
    # "BorderTransportMeans",
]

es.drop_columns(es_drop_columns)
es.factorize_columns(es_categorical_columns)
es.rate_condense(["DutyRegime"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE)
es.rate_condense(["DeclarationOfficeID"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE)
es.rate_condense(["BorderTransportMeans"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE)
es.rate_condense(["ExportationCountry"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE)
es.rate_condense(["OriginCountry"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE)

es.train_data.info()

es_corr = es.train_data.corr()
sns.heatmap(es_corr, annot=True, cmap=plt.cm.Reds)

# %%
# Project 2 Experiment, refactored
experiment_p2 = Experiment(train_data_raw, test_data_raw)
p2_drop_columns = [
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
    "IssueDateTime",
]
p2_categorical_columns = [
    "DeclarationOfficeID",
    "PaymentType",
    "BorderTransportMeans",
    "ExportationCountry",
]

experiment_p2.drop_columns(p2_drop_columns)
experiment_p2.factorize_columns(p2_categorical_columns)
experiment_p2.test_suite(verbose=True)

# %%
# Project 2 Experiment, refactored
experiment_p2 = Experiment(train_data_raw, test_data_raw)
p2_drop_columns = [
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
    "IssueDateTime",
]
p2_categorical_columns = [
    "DeclarationOfficeID",
    "PaymentType",
    "BorderTransportMeans",
    "ExportationCountry",
]

experiment_p2.drop_columns(p2_drop_columns)
experiment_p2.factorize_columns(p2_categorical_columns)
experiment_p2.test_suite(verbose=True)

# %%
e1 = Experiment(train_data_raw, test_data_raw)
e1_drop_columns = [
    "ID",
    "ProcessType",
    "Type",
    "DeclarerID",
    "ImporterID",
    "SellerID",
    "ExpressID",
    "DisplayIndicator",
    "IssueDateTime",
    "ClassificationID",
    # "AdValoremTaxBaseAmount(Won)",
    # "TotalGrossMassMeasure(KG)",
]
e1_categorical_columns = [
    "TransactionNature",
    # "DeclarationOfficeID",
    "PaymentType",
    # "BorderTransportMeans",
]

e1.drop_columns(e1_drop_columns)
e1.factorize_columns(e1_categorical_columns)
e1.rate_condense(["DutyRegime"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE)
e1.rate_condense(["DeclarationOfficeID"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE)
e1.rate_condense(["BorderTransportMeans"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE)
e1.rate_condense(
    ["ExportationCountry", "OriginCountry"], adj_amount=1, adj_mode=ADJ_MODE_BASERATE
)

e1.train_data.info()


corr = e1.train_data.corr()
sns.heatmap(corr, annot=True, fmt=".2f")

# %%
e1.evaluate_features(k=5)
res = e1.test_suite(verbose=True)

# %%
