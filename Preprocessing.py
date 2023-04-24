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
# Written as a synced .py file, executable as a notebook by the Jupytext extension.
#

# %%
# Basic imports and file paths

import numpy as np
import pandas as pd
import os

# import data
train_file = os.path.join(os.getcwd(), "data", "train.csv")
test_file = os.path.join(os.getcwd(), "data", "test.csv")

train_data_raw = pd.read_csv(train_file, parse_dates=["IssueDateTime"])
test_data_raw = pd.read_csv(test_file, parse_dates=["IssueDateTime"])



# %% [markdown]
# ### Data Exploration
#

# %%
train_data_raw.head()


# %%
train_data_raw.info()


# %%
# Question 1
# Also answerable from the info() method

for column in train_data_raw.columns:
    print(column, train_data_raw[column].isnull().sum())


# %%
# Question 2
train_subset_maritime = train_data_raw.loc[train_data_raw["BorderTransportMeans"] == 10]
train_subset_air = train_data_raw.loc[train_data_raw["BorderTransportMeans"] == 40]

maritime_count_total = train_subset_maritime.shape[0]
maritime_count_fake = sum(train_subset_maritime["Fake"])
air_count_total = train_subset_air.shape[0]
air_count_fake = sum(train_subset_air["Fake"])

print(
    f"Martime: {maritime_count_fake} out of {maritime_count_total} are fake. {maritime_count_fake/maritime_count_total*100:.2f}%"
)
print(
    f"Air: {air_count_fake} out of {air_count_total} are fake. {air_count_fake/air_count_total*100:.2f}%"
)


# %%
# Question 3
train_subset_icnharbor = train_data_raw.loc[train_data_raw["DeclarationOfficeID"] == 20]
train_subset_icnairport = train_data_raw.loc[
    train_data_raw["DeclarationOfficeID"] == 40
]

icnharbor_count_total = train_subset_icnharbor.shape[0]
icnharbor_count_fake = sum(train_subset_icnharbor["Fake"])
icnairport_count_total = train_subset_icnairport.shape[0]
icnairport_count_fake = sum(train_subset_icnairport["Fake"])

print(
    f"Incheon Harbor: {icnharbor_count_fake} out of {icnharbor_count_total} are fake. {icnharbor_count_fake/icnharbor_count_total*100:.2f}%"
)
print(
    f"Incheon Airport: {icnairport_count_fake} out of {icnairport_count_total} are fake. {icnairport_count_fake/icnairport_count_total*100:.2f}%"
)


# %%
# Question 4

# Below code is inefficient because it iterates over the whole dataset for each country.
# However, since the dataset is small, for the sake of clarity it is left as is.

exportation_country_list = train_data_raw["ExportationCountry"].unique()
fake_rate_bycountry = []

for country in exportation_country_list:
    subset_country = train_data_raw.loc[train_data_raw["ExportationCountry"] == country]
    fakes_country = sum(subset_country["Fake"])
    total_country = subset_country.shape[0]
    fake_rate_bycountry.append((country, fakes_country / total_country))

fake_rate_bycountry.sort(key=lambda x: x[1], reverse=True)

# print(fake_rate_bycountry)

countries_of_interest = ["IT", "US", "CN", "GB", "HK", "FR"]

for country, rate in fake_rate_bycountry:
    if country in countries_of_interest:
        print(f"{country}: {rate}")


# %%
# Question 5

# Modified read_csv while working on this problem to parse dates

fake_rate_bymonth = []
for month in range(1, 13):
    subset_month = train_data_raw.loc[train_data_raw["IssueDateTime"].dt.month == month]
    fakes_month = sum(subset_month["Fake"])
    total_month = subset_month.shape[0]
    fake_rate_bymonth.append((month, fakes_month / total_month))

fake_rate_bymonth.sort(key=lambda x: x[1], reverse=True)

for month, rate in fake_rate_bymonth:
    print(f"{month}: {rate}")


# %%
# Question 6

cols = ["TotalGrossMassMeasure(KG)", "AdValoremTaxBaseAmount(Won)"]
print(train_data_raw[cols[0]].corr(train_data_raw[cols[1]], method="pearson"))


# %%
# Question 7

cols = ["TaxRate", "Fake"]
print(train_data_raw[cols[0]].corr(train_data_raw[cols[1]], method="pearson"))

