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
# Holds its own copy of the dataframe, and can be reset to the original state
# Allows for easy experimentation with different preprocessing steps


def prep_step(func):
    def wrapper(*args, **kwargs):
        if not args[0].preprocessing_flag:
            print("Reenabling preprocessing.")
            args[0].reenable_preprocessing()
        return func(*args, **kwargs)

    return wrapper


def test_step(func):
    def wrapper(*args, **kwargs):
        if args[0].preprocessing_flag:
            print("Finalizing preprocessing.")
            args[0].finalize_preprocessing()
        return func(*args, **kwargs)

    return wrapper


class Experiment:
    def __init__(self, train_data_raw, test_data_raw):
        self.train_data_raw = train_data_raw
        self.test_data_raw = test_data_raw
        self.reset()

    def reset(self):
        train_data = self.train_data_raw.copy()
        test_data = self.test_data_raw.copy()

        # Join the two datasets for preprocessing
        train_data["data_type"] = "train"
        test_data["data_type"] = "test"
        test_data["Fake"] = pd.NA

        self.data = pd.concat([train_data, test_data], axis=0)
        self.preprocessing_flag = True
        self.newcol_count = 0

    def reenable_preprocessing(self):
        self.train_data["data_type"] = "train"
        self.train_data["ID"] = pd.NA

        self.test_data["data_type"] = "test"
        self.test_data["Fake"] = pd.NA
        self.test_data["ID"] = self.ID

        self.data = pd.concat([self.train_data, self.test_data], axis=0)
        self.preprocessing_flag = True

    def finalize_preprocessing(self):
        self.train_data = self.data[self.data["data_type"] == "train"].copy()
        self.test_data = self.data[self.data["data_type"] == "test"].copy()

        self.train_data.drop(columns=["data_type"], inplace=True)
        self.test_data.drop(columns=["data_type"], inplace=True)

        self.train_data["Fake"] = self.train_data["Fake"].astype(int)
        self.test_data.drop(columns=["Fake"], inplace=True)

        self.ID = self.test_data["ID"]
        self.train_data.drop(columns=["ID"], inplace=True)
        self.test_data.drop(columns=["ID"], inplace=True)
        self.preprocessing_flag = False

    @prep_step
    def drop_columns(self, columns):
        self.data.drop(columns=columns, inplace=True)

    @prep_step
    def dummy_columns(self, columns):
        self.data = pd.get_dummies(self.data, columns=columns)

    @prep_step
    def normalize_columns(self, columns):
        for column in columns:
            max_val = self.data[column].max()
            min_val = self.data[column].min()
            self.data[column] = (self.data[column] - min_val) / (max_val - min_val)

    @prep_step
    def factorize_columns(self, columns):
        for column in columns:
            self.data[column] = pd.factorize(self.data[column])[0]

    @test_step
    def predict(self, model=None, export=False):
        if model is None:
            model = self.best_model
        predictions = model.predict(self.test_data)
        results = pd.DataFrame({"ID": self.ID, "Fake": predictions.flatten()})
        if export:
            results.to_csv("20160840_submission.csv", index=False)
        return results


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


@prep_step
def rate_condense(self, columns, adj_amount=1, adj_mode=ADJ_MODE_BASERATE):
    train_data = self.data[self.data["data_type"] == "train"]
    rate_dict = defaultdict(lambda: [0, 0])
    for index, row in train_data.iterrows():
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
        r = pos / (neg + pos)

    newcol_name = "rate" + str(self.newcol_count)
    self.newcol_count += 1

    self.data[newcol_name] = self.data[columns].apply(lambda x: rates[tuple(x)], axis=1)
    self.data.drop(columns=columns, inplace=True)
    return newcol_name


@prep_step
def hash_condense(self, columns):
    newcol_name = "hash" + str(self.newcol_count)
    self.newcol_count += 1
    self.data[newcol_name] = self.data[columns].apply(lambda x: hash(tuple(x)), axis=1)
    self.data.drop(columns=columns, inplace=True)
    return newcol_name


Experiment.rate_condense = rate_condense
Experiment.hash_condense = hash_condense


# %%
@prep_step
def analyze_category(self, columns, high=0.8, low=0.05, min_count=10, ensure_both=True, verbose = False):
    if verbose:
        print("Analyzing category", columns)
    rate_dict = defaultdict(lambda: [0, 0])
    train_data = self.data[self.data["data_type"] == "train"]
    for index, row in train_data.iterrows():
        key = tuple(row[columns])
        if row["Fake"]:
            rate_dict[key][1] += 1
        else:
            rate_dict[key][0] += 1
    extreme = 0
    for key in rate_dict:
        neg, pos = rate_dict[key]
        r = pos / (neg + pos)
        if (
            (r > high or r < low)
            and neg + pos > min_count
            and (not ensure_both or (neg > 0 and pos > 0))
        ):
            extreme += 1
            # print(columns, f"::{key}: {neg} {pos}, rate: {r}")
    extreme_rate = extreme / len(rate_dict)
    if verbose:
        print(f"Extreme count: {extreme}")
        print(f"Total count: {len(rate_dict)}")
        print(f"Extreme rate: {extreme_rate}")
    return extreme_rate


@prep_step
def extract_extremes(self, column, high=0.8, low=0.05, min_count=10, ensure_both=True):
    rate_dict = defaultdict(lambda: [0, 0])
    train_data = self.data[self.data["data_type"] == "train"]
    for index, row in train_data.iterrows():
        key = row[column]
        if row["Fake"]:
            rate_dict[key][1] += 1
        else:
            rate_dict[key][0] += 1

    res = pd.DataFrame(columns=train_data.columns)
    extreme = 0
    for key in rate_dict:
        neg, pos = rate_dict[key]
        r = pos / (neg + pos)
        if (
            (r > high or r < low)
            and neg + pos > min_count
            and (not ensure_both or (neg > 0 and pos > 0))
        ):
            extreme += 1
            res = pd.concat([res, train_data[train_data[column] == key]])
    return res


Experiment.analyze_category = analyze_category
Experiment.extract_extremes = extract_extremes

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras


@test_step
def test_suite(
    self,
    knn_args=range(1, 10),
    rf_args=map(lambda x: 2**x, range(6, 12)),
    tf_flag=False,
    verbose=False,
):
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

    for k in knn_args:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(train_part_data, train_part_labels)
        knn_pred = knn.predict(test_part_data)
        score = accuracy_score(test_part_labels, knn_pred)
        if score > max_score:
            max_score = score
            self.best_model = knn
        if verbose:
            print(f"Accuracy for k={k}: {score:.3f}.")

    for estimators in rf_args:
        rf = RandomForestClassifier(n_estimators=estimators)
        rf.fit(train_part_data, train_part_labels)
        rf_pred = rf.predict(test_part_data)
        score = accuracy_score(test_part_labels, rf_pred)
        if score > max_score:
            max_score = score
            self.best_model = rf
        if verbose:
            print(f"Accuracy for n_estimators={estimators}: {score:.3f}.")

    if tf_flag:
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    64, activation="relu", input_shape=[len(train_part_data.keys())]
                ),
                keras.layers.Dense(64, activation="relu"),
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
    "ProcessType",
    "Type",
    "DeclarerID",
    "ImporterID",
    "SellerID",
    "ExpressID",
    "DisplayIndicator",
    "IssueDateTime",
    "ClassificationID",
]
es_categorical_columns = [
    "TransactionNature",
    "DeclarationOfficeID",
    "PaymentType",
    "BorderTransportMeans",
    "DutyRegime",
    "ExportationCountry",
    "OriginCountry",
]

es_normalize_columns = [
    "AdValoremTaxBaseAmount(Won)",
    "TotalGrossMassMeasure(KG)",
]
s1 = es.extract_extremes("DeclarerID")
s2 = es.extract_extremes("ImporterID")
s3 = es.extract_extremes("SellerID")
train_set = pd.concat([s1, s2, s3])
train_set.drop_duplicates(inplace=True)

train_set.drop(columns=es_drop_columns, inplace=True)
train_set.drop(columns=es_categorical_columns, inplace=True)
train_set.drop(columns=["data_type"], inplace=True)

# %%
# Project 2 Experiment, refactored
# Datetime also dropped in favor of random split
experiment_p2 = Experiment(train_data_raw, test_data_raw)
p2_drop_columns = [
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
experiment_p2.finalize_preprocessing()
experiment_p2.test_suite(verbose=True)

# %%
exp_p0 = Experiment(train_data_raw, test_data_raw)

# for col in exp_p0.data.columns:
#     exp_p0.analyze_category([col],high=0.4, low=0.1, verbose=True)

exp_p0.analyze_category(["DeclarationOfficeID","BorderTransportMeans"], high=0.4, low=0.1, verbose=True)

# %%
# Simple run with the most important features
exp_s1 = Experiment(train_data_raw, test_data_raw)
s1_drop_columns = [
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
    "ClassificationID",
    "TaxRate",
]
s1_categorical_columns = [
    "DeclarationOfficeID",
    "PaymentType",
    "BorderTransportMeans",
    "ExportationCountry",
]

exp_s1.drop_columns(s1_drop_columns)
exp_s1.drop_columns(s1_categorical_columns)
exp_s1.finalize_preprocessing()
exp_s1.test_suite(verbose=True)
exp_s1.predict(export=True)

# %%
exp_s2 = Experiment(train_data_raw, test_data_raw)
s2_categorical_columns = [
    "DeclarationOfficeID",
    "PaymentType",
    "BorderTransportMeans",
    "ExportationCountry",
    "OriginCountry",
    "ProcessType",
    "TransactionNature",
    "Type",
    "DeclarerID",
    "ImporterID",
    "SellerID",
    "ExpressID",
    "DutyRegime",
]
s2_drop_columns = [
    "DeclarationOfficeID",
    "PaymentType",
    "BorderTransportMeans",
    # "ExportationCountry",
    # "OriginCountry",
    "ProcessType",
    "TransactionNature",
    "Type",
    "DeclarerID",
    "ImporterID",
    "SellerID",
    "ExpressID",
    "DisplayIndicator",
    # "DutyRegime",
    "IssueDateTime",
    "ClassificationID",
    # "TaxRate",
    # "AdValoremTaxBaseAmount(Won)",
    # "TotalGrossMassMeasure(KG)",
]
s2_dummy_columns = [
    # "DeclarationOfficeID",
    # "PaymentType",
    # "BorderTransportMeans",
    # "ExportationCountry",
    # "OriginCountry",
    # "ProcessType",
    # "TransactionNature",
    # "Type",
    # "DeclarerID",
    # "ImporterID",
    # "SellerID",
    # "ExpressID",
    "DutyRegime",
]

exp_s2.factorize_columns(s2_categorical_columns)
exp_s2.dummy_columns(s2_dummy_columns)

s2_hashcol_1 = exp_s2.hash_condense(columns=["ExportationCountry", "OriginCountry"])
exp_s2.factorize_columns([s2_hashcol_1])
# exp_s2.dummy_columns([s2_hashcol_1])

# s2_hashcol_2 = exp_s2.hash_condense(columns=["DeclarationOfficeID", "BorderTransportMeans"])
# exp_s2.factorize_columns([s2_hashcol_2])

# exp_s2.data["AdValoremTaxBaseAmount(Won)"] = exp_s2.data[
#     "AdValoremTaxBaseAmount(Won)"
# ].apply(lambda x: x**2)
# exp_s2.data["TotalGrossMassMeasure(KG)"] = exp_s2.data[
#     "TotalGrossMassMeasure(KG)"
# ].apply(lambda x: x**2)

exp_s2.drop_columns(s2_drop_columns)
exp_s2.finalize_preprocessing()
exp_s2.test_suite(knn_args=[5, 7, 9], rf_args=[100, 200, 400, 1000], verbose=True)
exp_s2.predict(export=True)
