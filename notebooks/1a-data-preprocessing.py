# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: QRT-venv
#     language: python
#     name: qrt-venv
# ---

# %% [markdown]
# # Preprocessing
#
# - Set lower-accuracy datatypes for columns
# - Convert strings
# - Rename cols
# - Save as parquet

# %% editable=true slideshow={"slide_type": ""}
import tools

import numpy as np
import pandas as pd

# %%
train = pd.read_csv("../data/0-raw/X_train.csv")
test = pd.read_csv("../data/0-raw/X_test.csv")
y_train = pd.read_csv("../data/0-raw/y_train.csv")

# %% [markdown]
# # Data Types

# %%
print("=== y_train ===")
print(y_train.dtypes.to_string())

# %%
print("=== train dtypes ===")
print(train.dtypes.value_counts())
print(train.dtypes.to_string())

# %% [markdown]
# # Renames

# %%
COL_RENAMES = {
    "MEDIAN_DAILY_TURNOVER": "TURN",
    "ALLOCATION": "ALLOC",
    **{f"SIGNED_VOLUME_{i}": f"SVOL_{i}" for i in range(1, 21)},
}

for df in [train, test]:
    df.rename(columns=COL_RENAMES, inplace=True)

# %% [markdown]
# # Cleanup strings/categoricals

# %%
print(train[["ROW_ID", "TS", "ALLOC", "GROUP"]].head(10).to_string())
print(train[["ROW_ID", "TS", "ALLOC", "GROUP"]].tail(10).to_string())

# %%
for df in [train, test]:
    df["TS"] = df["TS"].str.removeprefix("DATE_").astype("uint32")
    df["ALLOC"] = df["ALLOC"].str.removeprefix("ALLOCATION_").astype("uint32")
    df["GROUP"] = df["GROUP"].astype("uint32")
    df["ROW_ID"] = df["ROW_ID"].astype("uint32")

# %%
print(train[["ROW_ID", "TS", "ALLOC", "GROUP"]].describe().to_string())

# %% [markdown]
# # Combine X and y

# %%
train["RET_0"] = y_train["target"]
test["RET_0"] = np.float32(np.nan)

# %% [markdown]
# # Save

# %%
tools.convert_types([train, test])

# %%
print("=== train dtypes ===")
print(train.dtypes.value_counts())
print(train.dtypes.to_string())

# %%
tools.clean("1a-preprocessed")
tools.save("1a-preprocessed/full", train, test, verbose=True)
