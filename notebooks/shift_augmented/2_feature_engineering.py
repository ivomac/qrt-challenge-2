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
# # Feature Engineering — Shift-Augmented
#
# **Purpose**: Generate derived features for the shift-augmented dataset. Unlike the
# main pipeline's ~4800 features, this uses a focused set of ~24 features optimized
# for the augmented data structure.
#
# **Feature families**:
# - **Row aggregates**: mean, sharpe, std for RET, SVOL, and VOLUME across time windows
# - **TS cross-sectional stats**: (TS, SHIFT)-level demean, percentile rank, and ratio
#   of row aggregates — computed on the full pool for consistent market context
# - **Alloc features**: fold-safe per-allocation sign bias, deviation, and sharpe,
#   fitted only on training fold then joined to both train and val
#
# Reads from `4c-combined/folds/`, writes to `4d-features/folds/`.

# %%
import tools

import numpy as np
import pandas as pd
import scipy.stats as st

import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)

# %%
N_SPLITS = 12


# %%
def add_row_features(X: pd.DataFrame) -> None:
    """Per-row lag aggregates. Modifies X in place."""
    ret = X[[f"RET_{i}" for i in range(1, 21)]].to_numpy(np.float32)
    svol = X[[f"SVOL_{i}" for i in range(1, 21)]].to_numpy(np.float32)
    vol = np.abs(svol)
    r1 = ret[:, 0]
    s1 = svol[:, 0]
    turn = X["TURN"].to_numpy(np.float32)
    n = len(X)

    def _mean(mat, a, b):
        return np.nanmean(mat[:, a - 1 : b], axis=1).astype(np.float32)

    def _std(mat, a, b):
        return np.nanstd(mat[:, a - 1 : b], axis=1).astype(np.float32)

    def _sharpe(mat, a, b):
        m = np.nanmean(mat[:, a - 1 : b], axis=1).astype(np.float32)
        s = np.nanstd(mat[:, a - 1 : b], axis=1).astype(np.float32)
        out = np.full(n, np.nan, dtype=np.float32)
        return np.divide(m, s, out=out, where=s > 0)

    def _div(num, den):
        out = np.full(n, np.nan, dtype=np.float32)
        return np.divide(num, den, out=out, where=den != 0)

    rm_1_4 = _mean(ret, 1, 4)

    X["row_mean_RET_1_4"] = rm_1_4
    X["row_sharpe_RET_1_4"] = _sharpe(ret, 1, 4)
    X["row_mean_RET_1_20"] = _mean(ret, 1, 20)
    X["row_sharpe_RET_1_20"] = _sharpe(ret, 1, 20)
    X["row_std_RET_1_20"] = _std(ret, 1, 20)
    X["row_mean_SVOL_1_4"] = _mean(svol, 1, 4)
    X["row_mean_SVOL_1_20"] = _mean(svol, 1, 20)
    X["row_mean_VOLUME_1_4"] = _mean(vol, 1, 4)
    X["sq_RET_1"] = (r1**2).astype(np.float32)
    X["prod_RET_1_SVOL_1"] = (r1 * s1).astype(np.float32)
    X["ratio_RET_1_TURN"] = _div(r1, turn)
    X["diff_RET_1_row_mean_RET_1_4"] = (r1 - rm_1_4).astype(np.float32)


def add_ts_features(X: pd.DataFrame) -> None:
    """(TS, SHIFT) cross-sectional stats. Must be called on the full combined pool."""
    grp = X.groupby(["TS", "SHIFT"], sort=False)
    ts_mean_turn = grp["TURN"].transform("mean").astype(np.float32)
    r1 = X["RET_1"].to_numpy(np.float32)
    tmt = ts_mean_turn.to_numpy(np.float32)
    out = np.full(len(X), np.nan, dtype=np.float32)

    X["ts_mean_TURN"] = ts_mean_turn
    X["ts_demean_RET_1"] = (X["RET_1"] - grp["RET_1"].transform("mean")).astype(np.float32)
    X["ts_pct_rank_RET_1"] = grp["RET_1"].rank(pct=True, na_option="keep").astype(np.float32)
    X["ts_pct_rank_TURN"] = grp["TURN"].rank(pct=True, na_option="keep").astype(np.float32)
    X["ts_pct_rank_row_mean_RET_1_4"] = (
        grp["row_mean_RET_1_4"].rank(pct=True, na_option="keep").astype(np.float32)
    )
    X["ts_pct_rank_row_sharpe_RET_1_4"] = (
        grp["row_sharpe_RET_1_4"].rank(pct=True, na_option="keep").astype(np.float32)
    )
    X["ts_pct_rank_SVOL_1"] = grp["SVOL_1"].rank(pct=True, na_option="keep").astype(np.float32)
    X["ratio_RET_1_ts_mean_TURN"] = np.divide(r1, tmt, out=out, where=tmt != 0)


def alloc_map(X_fold: pd.DataFrame) -> pd.DataFrame:
    """Fold-safe per-ALLOC features fitted on real-target rows only."""
    rows = X_fold[X_fold["RET_0"].notna()]
    grp = rows.groupby("ALLOC")

    agg = grp["RET_1"].agg(
        pos=lambda s: (s.dropna() > 0).sum(),
        total=lambda s: s.dropna().count(),
    )
    dev_vals, bias_vals = [], []
    for n_pos, n_tot in zip(agg["pos"].astype(int), agg["total"].astype(int)):
        if n_tot <= 1:
            dev_vals.append(np.float32(0.0))
            bias_vals.append(np.float32(0.0))
        else:
            d = n_pos / n_tot - 0.5
            p = st.binomtest(int(n_pos), int(n_tot), p=0.5).pvalue
            dev_vals.append(np.float32(d))
            bias_vals.append(np.float32(np.sign(d) * np.abs(np.log10(max(float(p), 1e-300)))))

    alloc_m = grp["row_mean_RET_1_4"].mean().astype(np.float32)
    alloc_s = grp["row_mean_RET_1_4"].std().astype(np.float32)
    sharpe = np.full(len(alloc_m), np.nan, dtype=np.float32)
    np.divide(alloc_m.to_numpy(), alloc_s.to_numpy(), out=sharpe, where=alloc_s.to_numpy() > 0)

    return pd.DataFrame(
        {
            "alloc_dev_RET_1": np.array(dev_vals, dtype=np.float32),
            "alloc_bias_RET_1": np.array(bias_vals, dtype=np.float32),
            "alloc_sharpe_row_mean_RET_1_4": sharpe,
        },
        index=agg.index,
    )


# %%
def process(X_fold: pd.DataFrame, X_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add all features to fold+val combined pool, return split."""
    n = len(X_fold)
    X = pd.concat([X_fold, X_val], ignore_index=True)
    add_row_features(X)
    add_ts_features(X)
    # alloc_map fitted on fold portion after row features are present
    am = alloc_map(X.iloc[:n].copy())
    X = X.join(am, on="ALLOC")
    return X.iloc[:n].copy(), X.iloc[n:].copy()


# %%
tools.clean("4d-features")

train, test = tools.load("4c-combined/full")
X_train_out, X_test_out = process(train, test)
tools.save("4d-features/full", X_train_out, X_test_out)
n_new = X_train_out.shape[1] - train.shape[1]
print(
    f"full: train={len(X_train_out):,}  test={len(X_test_out):,}  +{n_new} new cols  total={X_train_out.shape[1]}"
)
del train, test, X_train_out, X_test_out

# %%
for k in range(N_SPLITS):
    X_fold, X_val = tools.load(f"4c-combined/folds/{k}")
    X_fold_out, X_val_out = process(X_fold, X_val)
    tools.save(f"4d-features/folds/{k}", X_fold_out, X_val_out)
    print(
        f"fold {k:2d}: train={len(X_fold_out):,}  val={len(X_val_out):,}  cols={X_fold_out.shape[1]}"
    )
    del X_fold, X_val, X_fold_out, X_val_out
