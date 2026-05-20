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
# # Shift-Augmented Data Generation
#
# **Purpose**: A separate experiment exploring whether time-shift augmentation improves
# model performance. The idea is that each row's 20-day feature window can be shifted
# backwards in time to create synthetic training examples, multiplying the dataset size.
#
# **Approach**:
# - Concatenates train and test sets, then creates 12 GroupKFold splits by TS
# - For each fold's training data, generates 7 shifted copies (original + shifts 1-7)
#   where RET_* and SVOL_* columns are shifted so RET_k at shift s becomes RET_{k-s}
# - TURNOVER is set to NaN for shifted rows (only meaningful at shift=0)
# - Validation data is kept unshifted (SHIFT=0)
# - Outputs to `4c-combined/folds/` and `4c-combined/full/`
#
# **Conclusion**: This experiment was an alternate approach that did not outperform the
# main pipeline. Kept for reference as a data augmentation technique.

# %%
import tools

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# %%
N_SPLITS = 12
N_SHIFTS = 7

RET_COLS = [f"RET_{i}" for i in range(0, 21)]
SVOL_COLS = [f"SVOL_{i}" for i in range(1, 21)]

# %%
train, test = tools.load("1a-preprocessed/full")


# %%
def shift_augment(X, n_shifts):
    ret = X[RET_COLS].to_numpy(np.float32)
    svol = X[SVOL_COLS].to_numpy(np.float32)
    turn = X["TURN"].to_numpy(np.float32)
    n, n_ret, n_svol = len(X), len(RET_COLS), len(SVOL_COLS)
    ret_rows, svol_rows, turn_rows, shf_rows = [], [], [], []
    for s in range(n_shifts + 1):
        nr = np.full((n, n_ret), np.nan, dtype=np.float32)
        ns = np.full((n, n_svol), np.nan, dtype=np.float32)
        nr[:, : n_ret - s] = ret[:, s:]
        ns[:, : n_svol - s] = svol[:, s:]
        ret_rows.append(nr)
        svol_rows.append(ns)
        turn_rows.append(turn if s == 0 else np.full(n, np.nan, dtype=np.float32))
        shf_rows.append(np.full(n, s, dtype=np.uint32))
    df = pd.concat([X] * (n_shifts + 1), ignore_index=True)
    df[RET_COLS] = np.vstack(ret_rows)
    df[SVOL_COLS] = np.vstack(svol_rows)
    df["TURN"] = np.concatenate(turn_rows)
    df["SHIFT"] = np.concatenate(shf_rows)
    # Test rows (RET_0=NaN) contribute augmented examples at s>=1
    # because their RET_s is a real lag value, not a target to be withheld.
    mask = ~np.isnan(df["RET_0"]) & ~np.isnan(df["RET_1"])
    return df[mask].reset_index(drop=True)


# %%
tools.clean("4c-combined")
gkf = GroupKFold(n_splits=N_SPLITS)

train = pd.concat([train, test])

for k, (tr, va) in enumerate(gkf.split(train, groups=train["TS"].values)):
    X_fold, X_val = train.iloc[tr], train.iloc[va]
    X_val = X_val[~np.isnan(X_val["RET_0"])]
    n_tr_ts, n_va_ts = X_fold["TS"].nunique(), X_val["TS"].nunique()
    X_fold = shift_augment(X_fold, N_SHIFTS)
    X_val = X_val.assign(SHIFT=np.uint32(0))
    tools.save(f"4c-combined/folds/{k}", X_fold, X_val)
    print(f"fold {k:2d}: train={len(X_fold)} ({n_tr_ts} TS) -- val={len(X_val)} ({n_va_ts} TS)")

n_tr_ts, n_te_ts = train["TS"].nunique(), test["TS"].nunique()
train = shift_augment(train, N_SHIFTS)
test = test.assign(SHIFT=np.uint32(0))
tools.save("4c-combined/full", train, test)
print(f"full   : train={len(train)} ({n_tr_ts} TS) -- tst={len(X_val)} ({n_te_ts} TS)")
