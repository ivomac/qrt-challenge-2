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
# # Post-analysis Processing
#
# ## Purpose
# Clean outliers and normalize base features before feature engineering.
#
# ## Key Insights
# - ALLOC 14 and 46 have only 19 rows each (2b) — both dropped.
# - Clip bounds from 2f Q-Q analysis: RET ±0.01, SVOL ±7.5, TURN [1e-4, 1.6].
# - Target clipped at ±0.01 (~3.2σ) — trims extreme tails while preserving
#   central signal. Tight clip chosen to bound Huber loss influence of outliers.
# - Normalize by clip half-range → values ~[-1, 1] for LightGBM numerical stability.
# - Saves to `3a-postprocessed/full`.

# %%
import tools

# %%
train, test = tools.load("1a-preprocessed/full")

# %% [markdown]
# ## Drop ALLOC 14 and 46

# %%
mask = train["ALLOC"].isin([14, 46])
print(f"Rows to drop: {mask.sum()}")
train = train[~mask].copy()
print(f"train after drop: {train.shape}")

# %% [markdown]
# ## Clip bounds (from 2f analysis)

# %%
# Determined from Q-Q plots and density-difference analysis in 2f
RET_COLS = [f"RET_{i}" for i in range(0, 21)]
SVOL_COLS = [f"SVOL_{i}" for i in range(1, 21)]

CLIP_BOUNDS = {}
for c in RET_COLS:
    CLIP_BOUNDS[c] = (-0.01, 0.01, 0.01)
for c in SVOL_COLS:
    CLIP_BOUNDS[c] = (-7.5, 7.5, 7.5)
CLIP_BOUNDS["TURN"] = (1e-4, 1.6, 1.6)

for col, (lo, hi, div) in CLIP_BOUNDS.items():
    if col in train.columns:
        train[col] = (train[col].clip(lo, hi) / div).astype("float32")

for col, (lo, hi, div) in CLIP_BOUNDS.items():
    if col in test.columns:
        test[col] = (test[col].clip(lo, hi) / div).astype("float32")

# %% [markdown]
# ## Save

# %%
tools.clean("3a-postprocessed")

tools.convert_types([train, test])
tools.save("3a-postprocessed/full", train, test, verbose=True)
