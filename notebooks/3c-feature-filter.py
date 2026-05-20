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
# # Feature Filtering
#
# Sort features by abs(rho target) (global Spearman vs target), iterate and keep if abs(rho target) < CORR_THRESHOLD with all already-kept features.

# %%
import tools

import gc

import numpy as np
import pandas as pd

from scipy.stats import rankdata

# %%
N_SPLITS = 12
CORR_THRESHOLD = 0.95

# %% [markdown]
# ## Load

# %%
train, test = tools.load("3b-engineered/full")

# %%
ID_COLS = {"TS", "ALLOC", "GROUP", "ROW_ID", "RET_0"}
NUMBER_COLS = list(set(train.select_dtypes(include="number").columns) - ID_COLS)

# %% [markdown]
# ## Spearman: feature ↔ target and feature ↔ feature

# %%
n_rows = len(train)
n_cols = len(NUMBER_COLS)
row_batch = 50_000
# In-RAM rank buffer (~8.5 GB). A disk memmap is fatal here: writing whole
# columns (mm[:, j]) into a row-major file is a strided write that re-flushes
# every page (~850x write amplification). RAM makes strided column writes free.
# Peak stays ~12 GB because train columns are freed as they are ranked and no
# separate mask array is kept (missingness is read back as isnan).

# --- Pass 1: rank each column to (0, 1] (NaN kept for missing); accumulate
#     count/sum/sumsq from the in-memory column so no extra read pass is needed ---
mm = np.empty((n_rows, n_cols), dtype=np.float32)
n_eff = np.zeros(n_cols, dtype=np.float64)
col_sum = np.zeros(n_cols, dtype=np.float64)
col_sumsq = np.zeros(n_cols, dtype=np.float64)
for j, col in enumerate(NUMBER_COLS):
    col_vals = train[col].values.astype(np.float32)
    valid = ~np.isnan(col_vals)
    nv = int(valid.sum())
    ranked = np.full(n_rows, np.nan, dtype=np.float32)
    r = rankdata(col_vals[valid], method="average") / max(nv, 1)  # scale to (0,1] for f32 safety
    ranked[valid] = r.astype(np.float32)
    mm[:, j] = ranked
    n_eff[j] = nv
    col_sum[j] = r.sum()
    col_sumsq[j] = (r * r).sum()
    if (j + 1) % 500 == 0:
        print(f"ranked {j + 1}/{n_cols} cols")
print(f"ranked all {n_cols} cols")

# Free train columns no longer needed
for col in NUMBER_COLS:
    del train[col]

n_eff_safe = np.maximum(n_eff, 1)
rank_mean = (col_sum / n_eff_safe).astype(np.float32)
rank_std = np.sqrt(np.maximum(col_sumsq / n_eff_safe - (col_sum / n_eff_safe) ** 2, 0)).astype(
    np.float32
)

# target ranks (scaled), centered; standardized at the end
y_ranked = (rankdata(train["RET_0"].values, method="average") / n_rows).astype(np.float32)
y_centered = (y_ranked - y_ranked.mean()).astype(np.float32)
y_std = float(y_ranked.std())

# --- Pass 2: one read accumulates n_pair (BLAS float, not int matmul), the
#     centered cross-products XtX, and the target dot product together ---
n_pair = np.zeros((n_cols, n_cols), dtype=np.float64)
XtX = np.zeros((n_cols, n_cols), dtype=np.float64)
Xty = np.zeros(n_cols, dtype=np.float64)
for start in range(0, n_rows, row_batch):
    batch = mm[start : start + row_batch]
    nanmask = np.isnan(batch)
    mask = (~nanmask).astype(np.float32)
    centered = np.where(nanmask, np.float32(0.0), batch - rank_mean)
    n_pair += mask.T @ mask
    XtX += centered.T @ centered
    Xty += centered.T @ y_centered[start : start + batch.shape[0]]
    print(f"corr sweep: {start}/{n_rows}")

# --- Feature-feature correlation (pairwise-complete) ---
n_pair_safe = np.maximum(n_pair, 1.0)
std_outer = np.outer(np.maximum(rank_std, 1e-12), np.maximum(rank_std, 1e-12))
corr_arr = np.clip((XtX / n_pair_safe) / std_outer, -1.0, 1.0)
corr_matrix = pd.DataFrame(corr_arr, index=NUMBER_COLS, columns=NUMBER_COLS)

# --- Feature-target correlation (pairwise-complete with target) ---
target_corr = (Xty / n_eff_safe) / (np.maximum(rank_std, 1e-12) * max(y_std, 1e-12))
target_corr[n_eff < 30] = np.nan
corr_target = pd.Series(target_corr.astype(np.float32), index=NUMBER_COLS, name="target")

# %%
abs_corr_target = corr_target.abs().sort_values(ascending=False)

print(
    abs_corr_target.head(20)
    .to_frame(name="Correlation")
    .to_string(formatters={"Correlation": "{:+.4f}".format})
)

# %%
n_pairs = len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2
iu = np.triu_indices(len(corr_matrix.columns), k=1)
n_high = (np.abs(corr_matrix.values[iu]) >= CORR_THRESHOLD).sum()
print(f"Pairs |ρ| ≥ {CORR_THRESHOLD}: {n_high} / {n_pairs} ({n_high / n_pairs:.2%})")

# %% [markdown]
# ## Greedy redundancy removal

# %%
corr_arr = np.abs(corr_matrix.values)
cols = list(corr_matrix.columns)
col_to_idx = {c: i for i, c in enumerate(cols)}
M = len(cols)
keep_mask = np.ones(M, dtype=bool)
drop_reason = {}

for col in abs_corr_target.index:
    i = col_to_idx[col]

    if not keep_mask[i]:
        continue

    correlated_indices = np.where(keep_mask & (corr_arr[i] >= CORR_THRESHOLD))[0]
    correlated_indices = correlated_indices[correlated_indices != i]

    for j in correlated_indices:
        keep_mask[j] = False
        drop_reason[cols[j]] = f"|ρ|={corr_arr[i, j]:.3f} with {col}"

kept_cols = {c for c, k in zip(cols, keep_mask) if k}

print(f"Original:            {M}")
print(f"Correlation-dropped: {M - len(kept_cols)}")
print(f"Final:               {len(kept_cols)}")

kept_cols = list(kept_cols | ID_COLS)

# %% [markdown]
# ## Final feature set

# %%
del mm, corr_matrix, corr_arr, abs_corr_target, corr_target
gc.collect()

tools.clean("3c-filtered/full")
train, test = tools.load_columns("3b-engineered/full", columns=kept_cols)

tools.save("3c-filtered/full", train, test)

del train, test
gc.collect()

# %% [markdown]
# ## Apply same drops to folds

# %%
tools.clean("3c-filtered/folds")

for k in range(N_SPLITS):
    train, test = tools.load_columns(f"3b-engineered/folds/{k}", columns=kept_cols)

    tools.save(f"3c-filtered/folds/{k}", train, test)
    del train, test
    gc.collect()
    print(f"Fold {k} done")
