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
# # Time reverse engineering
#
# The data's time column is anonymized and shuffled. In this notebook I check whether it is possible to reconstruct the time order by matching returns of rows of the same allocation with a shift between them.
#
# I try this on a random row, and since there are no matches, I think it's impossible to reverse engineer time.
#
# This is also useful to know: NO overlap between days. Each row is entirely new data.

# %%
import tools

import numpy as np

# %%
train, _ = tools.load("1a-preprocessed/full")

# %%
RET_COLS = [f"RET_{i}" for i in range(20, 0, -1)]  # RET_20 .. RET_1

# Find rows with no NaN across RET columns
valid = train[RET_COLS].notna().all(axis=1)
valid_idx = train.index[valid]
print(f"Rows with no NaN in RET cols: {len(valid_idx)} / {len(train)}")

# Pick one at random
rng = np.random.default_rng(42)
anchor_idx = rng.choice(valid_idx)
anchor = train.loc[anchor_idx]
anchor_alloc = anchor["ALLOC"]
anchor_ret = anchor[RET_COLS].values.astype(float)

print(f"Anchor row index: {anchor_idx}")
print(f"Anchor ALLOC: {anchor_alloc}")
print(f"Anchor RETs: {anchor_ret}")

# %%
EPSILON = 1e-7
OVERLAP_LEN = 6
SHIFT = 1

same_alloc = train[(train["ALLOC"] == anchor_alloc) & (train.index != anchor_idx) & valid]
print(f"Other rows from same ALLOC (no NaN): {len(same_alloc)}")

matches = []
for other_idx, other in same_alloc.iterrows():
    other_ret = other[RET_COLS].values
    diff = np.abs(other_ret[:OVERLAP_LEN] - anchor_ret[SHIFT : SHIFT + OVERLAP_LEN])
    if (diff < EPSILON).all():
        matches.append((other_idx, diff.max()))

print(f"\nMatches found: {len(matches)}")

# %%
# Try all possible shifts
print("Shifts with any match:")
for shift in range(0, 10):
    overlap = 6
    count = 0
    max_err = 0.0
    for other_idx, other in same_alloc.iterrows():
        other_ret = other[RET_COLS].values
        diff = np.abs(other_ret[:overlap] - anchor_ret[shift : shift + overlap])
        if (diff < EPSILON).all():
            count += 1
            max_err = max(max_err, diff.max())
    if count > 0:
        print(
            f"  shift={shift:2d} (overlap={overlap:2d}): {count:4d} matches, max_err={max_err:.2e}"
        )
