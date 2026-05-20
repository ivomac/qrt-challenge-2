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
# # Feature Correlations & Mutual Information
#
# - RET_1 is by far the most important feature
# - TURNOVER follows
# - then "~1 week ago" is also important (RET_7,8,9)
# - SVOL_11 is most important volume (?? noise?)
# - per-ALLOC target mean is stable

# %%
import tools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer


# %%
train, test = tools.load("1a-preprocessed/full")

# %%
target = train["RET_0"]
print(f"train: {train.shape}")
print("RET_0 in train")
print(
    f"Target  mean={target.mean():.3e}  std={target.std():.3e}  "
    f"positive_rate={(target > 0).mean():.2%}"
)

# %% [markdown]
# ## Feature groups

# %%
ID_COLS = ["TS", "ALLOC", "GROUP"]
RET_COLS = sorted(
    [c for c in train.columns if c.startswith("RET_")], key=lambda x: int(x.split("_")[1])
)
SVOL_COLS = sorted(
    [c for c in train.columns if c.startswith("SVOL_")], key=lambda x: int(x.split("_")[1])
)
OTHER_NUM = ["TURN"]
NUMERIC_COLS = RET_COLS + SVOL_COLS + OTHER_NUM

X_num = train[NUMERIC_COLS]

# %% [markdown]
# ## Spearman correlations with target

# %%
spearman_series = X_num.corrwith(target, method="spearman")
spearman_series.name = "spearman_rho"

# Per-feature effective n (pairwise-complete with target)
n_eff = X_num.notna().sum()
n_eff.name = "n_eff"
theo_se = 1 / np.sqrt(n_eff - 1)

# %%
spearman = spearman_series.to_frame(name="rho")
spearman["abs_rho"] = spearman["rho"].abs()
spearman = spearman.sort_values(by="abs_rho", ascending=False)
print(spearman.to_string(float_format=lambda x: f"{x:.4f}"))
print()
print(
    f"Effective n range: [{n_eff.min()}, {n_eff.max()}]  ({n_eff.min() / n_eff.max():.0%} of max)"
)
print(f"Theoretical SE range: [{theo_se.min():.6f}, {theo_se.max():.6f}]")

# %% [markdown]
# ### Null distribution
#
# Under the null, Spearman rho ~ N(0, 1/(n-1)).  Per-feature n varies
# due to NaN patterns (pairwise-complete with target).

# %%
spearman = spearman_series.sort_values(ascending=False).to_frame(name="rho")
features = spearman.index
rhos = spearman["rho"].values

theo_band = 2.576 * theo_se.loc[features].values

fig, ax = plt.subplots(1, 1, figsize=(6, 8))

colors = ["steelblue" if v > 0 else "darkorange" for v in rhos]
ax.barh(range(len(features)), rhos, xerr=theo_band, color=colors, capsize=2)

ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=8)
ax.invert_yaxis()
ax.set_title("Spearman rho vs target")
ax.axvline(0, color="gray", linewidth=0.5)

plt.show(block=False)

# %%
null_table = pd.DataFrame(
    {
        "rho": spearman_series,
        "|rho|": spearman_series.abs(),
        "n_eff": n_eff,
        "theo_lo": -2.576 * theo_se,
        "theo_hi": +2.576 * theo_se,
    }
)
null_table["significant"] = (null_table["rho"] < null_table["theo_lo"]) | (
    null_table["rho"] > null_table["theo_hi"]
)
null_table = null_table.sort_values("|rho|", ascending=False).drop(columns="|rho|")

print(null_table.to_string(float_format=lambda x: f"{x:+.4f}"))

# %% [markdown]
# ## Feature-feature correlations
#
# Per-pair null SE varies with NaN (pairwise-complete n).
# CI bands use +/-2.576 * SE where SE = 1 / sqrt(n_pair - 1).

# %%
notna_mat = X_num.notna().values.astype(int)
n_pair = pd.DataFrame(notna_mat.T @ notna_mat, index=NUMERIC_COLS, columns=NUMERIC_COLS)
se_pair = 1 / np.sqrt(np.maximum(n_pair - 1, 1))

# %% [markdown]
# ### RET_* lag-decay
#
# For each lag k, pool all pairs (RET_i, RET_{i+k}) and show mean rho.

# %%
corr_ret = X_num[RET_COLS].corr(method="spearman")
n_ret = len(RET_COLS)

lag_ret = {}
for k in range(1, n_ret):
    rhos = []
    ses = []
    for i in range(n_ret - k):
        j = i + k
        pair = (RET_COLS[i], RET_COLS[j])
        rhos.append(corr_ret.loc[pair])
        ses.append(se_pair.loc[pair])
    lag_ret[k] = {
        "rho_mean": np.mean(rhos),
        "se_mean": np.mean(ses),
        "n_pairs": len(rhos),
    }

lag_ret_df = pd.DataFrame(lag_ret).T
lag_ret_df["ci"] = 2.576 * lag_ret_df["se_mean"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(
    lag_ret_df.index,
    lag_ret_df["rho_mean"],
    yerr=lag_ret_df["ci"],
    fmt="o-",
    color="steelblue",
    capsize=3,
)
ax.set_xlabel("lag k")
ax.set_ylabel("Spearman rho")
ax.set_title("RET_* lag-decay (mean rho +/- 99% CI)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.grid(True, alpha=0.3)
plt.show(block=False)

# %% [markdown]
# ### SVOL_* lag-decay

# %%
corr_svol = X_num[SVOL_COLS].corr(method="spearman")
n_svol = len(SVOL_COLS)

lag_svol = {}
for k in range(1, n_svol):
    rhos = []
    ses = []
    for i in range(n_svol - k):
        j = i + k
        pair = (SVOL_COLS[i], SVOL_COLS[j])
        rhos.append(corr_svol.loc[pair])
        ses.append(se_pair.loc[pair])
    lag_svol[k] = {
        "rho_mean": np.mean(rhos),
        "se_mean": np.mean(ses),
        "n_pairs": len(rhos),
    }

lag_svol_df = pd.DataFrame(lag_svol).T
lag_svol_df["ci"] = 2.576 * lag_svol_df["se_mean"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(
    lag_svol_df.index,
    lag_svol_df["rho_mean"],
    yerr=lag_svol_df["ci"],
    fmt="o-",
    color="darkorange",
    capsize=3,
)
ax.set_xlabel("lag k")
ax.set_ylabel("Spearman rho")
ax.set_title("SVOL_* lag-decay (mean rho +/- 99% CI)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.grid(True, alpha=0.3)
plt.show(block=False)

# %% [markdown]
# ### TURN vs RET_* and SVOL_*

# %%
turn_ret = X_num[RET_COLS].corrwith(X_num["TURN"], method="spearman")
turn_ret_se = se_pair.loc["TURN", RET_COLS]
turn_ret_ci = 2.576 * turn_ret_se.values

turn_svol = X_num[SVOL_COLS].corrwith(X_num["TURN"], method="spearman")
turn_svol_se = se_pair.loc["TURN", SVOL_COLS]
turn_svol_ci = 2.576 * turn_svol_se.values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, rhos, cis, labels, color, title in [
    (axes[0], turn_ret, turn_ret_ci, turn_ret.index, "steelblue", "TURN vs RET_*"),
    (axes[1], turn_svol, turn_svol_ci, turn_svol.index, "darkorange", "TURN vs SVOL_*"),
]:
    colors = [color if v > 0 else "#d62728" for v in rhos.values]
    ax.barh(range(len(labels)), rhos.values, xerr=cis, color=colors, capsize=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.axvline(0, color="gray", linewidth=0.5)

plt.show(block=False)

# %% [markdown]
# ## Mutual information with target
#
# Null distribution from permutation test: shuffle target 100x,
# recompute MI.  CI bands are 1st/99th percentiles of the null.

# %%
N_PERMS = 100
N_BINS = 30

target_binary = (target > 0).astype(int)
print(f"Target positive rate: {target_binary.mean():.1%}")

# Compute MI per-feature using only non-NaN rows for that feature.
# This avoids median-fill distortion for high-NaN features like SVOL_1.
discretizer = KBinsDiscretizer(
    n_bins=N_BINS,
    encode="ordinal",
    strategy="quantile",
    quantile_method="averaged_inverted_cdf",
)

mi_binary_values = np.full(len(NUMERIC_COLS), np.nan)
rng = np.random.default_rng(42)

for j, col in enumerate(NUMERIC_COLS):
    valid_mask = X_num[col].notna().values
    n_valid = valid_mask.sum()
    if n_valid < 100:
        continue
    X_col = X_num.loc[valid_mask, [col]].values
    X_disc_col = discretizer.fit_transform(X_col).astype(int).ravel()
    mi_binary_values[j] = mutual_info_classif(
        X_disc_col.reshape(-1, 1),
        target_binary[valid_mask],
        discrete_features=True,
        random_state=42,
    )[0]

mi_binary_series = pd.Series(mi_binary_values, index=NUMERIC_COLS, name="MI_binary")
mi_binary_series = mi_binary_series.sort_values(ascending=False)

# Permutation null (per-feature, pairwise-complete)
null_mi_bin = np.full((N_PERMS, len(NUMERIC_COLS)), np.nan)
for i in range(N_PERMS):
    y_perm = rng.permutation(target_binary)
    for j, col in enumerate(NUMERIC_COLS):
        valid_mask = X_num[col].notna().values
        if valid_mask.sum() < 100:
            continue
        X_col = X_num.loc[valid_mask, [col]].values
        X_disc_col = discretizer.fit_transform(X_col).astype(int).ravel()
        null_mi_bin[i, j] = mutual_info_classif(
            X_disc_col.reshape(-1, 1),
            y_perm[valid_mask],
            discrete_features=True,
            random_state=rng.integers(0, 2**31),
        )[0]

null_bin_lo = pd.Series(
    np.nanpercentile(null_mi_bin, 1, axis=0),
    index=NUMERIC_COLS,
)
null_bin_hi = pd.Series(
    np.nanpercentile(null_mi_bin, 99, axis=0),
    index=NUMERIC_COLS,
)

features = mi_binary_series.index
fig, ax = plt.subplots(1, 1, figsize=(6, 8))

colors = ["steelblue" if v > null_bin_hi[f] else "darkorange" for f, v in mi_binary_series.items()]
ax.barh(
    range(len(features)),
    mi_binary_series.values,
    xerr=null_bin_hi.loc[features].values,
    color=colors,
    capsize=2,
)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=8)
ax.invert_yaxis()
ax.set_title("Binary MI (target sign)")
plt.show(block=False)


# %% [markdown]
# ### MI ranking comparison

# %%
mi_table = pd.DataFrame(
    {
        "MI_bin": mi_binary_series,
        "MI_bin_null_hi": null_bin_hi,
    }
)
mi_table["bin_sig"] = mi_table["MI_bin"] > mi_table["MI_bin_null_hi"]
mi_table = mi_table.sort_values("MI_bin", ascending=False)

# %%
spearman_ci = (2.576 * theo_se).loc[mi_binary_series.index].values
ranking_df = pd.DataFrame(
    {
        "spearman_rho": spearman_series,
        "spearman_rank": spearman_series.abs().rank(ascending=False),
        "MI_binary": mi_binary_series,
        "MI_binary_rank": mi_binary_series.rank(ascending=False),
    }
)
ranking_df["avg_rank"] = ranking_df[["spearman_rank", "MI_binary_rank"]].mean(axis=1)

print("=== Ranking by average rank ===")
print(
    ranking_df.sort_values("avg_rank").to_string(
        float_format=lambda x: f"{x:.2f}" if abs(x) >= 0.01 else f"{x:.2e}",
    )
)

# %%
# Scatter: abs Spearman rho vs MI with error bars (log-log)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

x = ranking_df["spearman_rho"].abs().values
y = ranking_df["MI_binary"].values
xerr = spearman_ci
yerr = null_bin_hi.loc[ranking_df.index].values

ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", alpha=0.5, markersize=5, capsize=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("|Spearman rho|")
ax.set_ylabel("MI binary")
ax.set_title("|Spearman rho| vs MI (binary)")

plt.show(block=False)

# %% [markdown]
# ## Does the marginal signal carry out of sample?
#
# A global feature->target rho can be an in-sample artifact, like the per-alloc
# bias in 2b. Test has no target, so: (1) split-half stability of feature->target
# rho within train; (2) feature->RET_1 rho train-vs-test (RET_1 is the strongest
# target proxy available on test).

# %%
# (1) split-half feature->target stability within train
rng = np.random.default_rng(0)
perm = rng.permutation(len(X_num))
a_idx, b_idx = perm[: len(perm) // 2], perm[len(perm) // 2 :]
rho_a = X_num.iloc[a_idx].corrwith(target.iloc[a_idx], method="spearman")
rho_b = X_num.iloc[b_idx].corrwith(target.iloc[b_idx], method="spearman")

stab = pd.DataFrame({"rho_full": spearman_series, "rho_A": rho_a, "rho_B": rho_b})
stab["sign_agree"] = np.sign(stab["rho_A"]) == np.sign(stab["rho_B"])
print("=== Split-half feature->target stability (within train) ===")
print(f"corr(rho_A, rho_B) across features: {stab['rho_A'].corr(stab['rho_B']):.3f}")
print(f"sign agreement: {int(stab['sign_agree'].sum())}/{len(stab)}")
print(
    stab.reindex(spearman_series.abs().sort_values(ascending=False).index)
    .head(15)
    .to_string(float_format=lambda x: f"{x:+.4f}")
)

# %%
# (2) feature->RET_1 rho: train vs test (RET_1 = target proxy available on test)
proxy_cols = [c for c in NUMERIC_COLS if c != "RET_1"]
tr_r1 = train[proxy_cols].corrwith(train["RET_1"], method="spearman")
te_r1 = test[proxy_cols].corrwith(test["RET_1"], method="spearman")

# how well does the RET_1 proxy track the real target signal on train?
proxy_track = spearman_series[proxy_cols].corr(tr_r1)
print("\n=== feature->RET_1 carry to test ===")
print(f"corr(feature->target, feature->RET_1) on train (proxy validity): {proxy_track:.3f}")

carry = pd.DataFrame(
    {"vs_target_train": spearman_series[proxy_cols], "RET1_train": tr_r1, "RET1_test": te_r1}
)
carry["sign_agree"] = np.sign(carry["RET1_train"]) == np.sign(carry["RET1_test"])
print(
    f"corr(RET1_train, RET1_test) across features: {carry['RET1_train'].corr(carry['RET1_test']):.3f}"
)
print(f"sign agreement train vs test: {int(carry['sign_agree'].sum())}/{len(carry)}")
print(
    carry.reindex(tr_r1.abs().sort_values(ascending=False).index)
    .head(15)
    .to_string(float_format=lambda x: f"{x:+.4f}")
)

# %% [markdown]
# ## Fair train-vs-test via a one-day-shifted window
#
# The target is the next-day return ("RET_0" in the sequence ... RET_2, RET_1,
# RET_0). Treat RET_1 as an anchor next-day return whose history is RET_2..RET_20:
# that is the real (history -> next-return) problem shifted one day, and RET_1 is
# observed in BOTH train and test. So the lag-1 autocorrelation RET_{k+1}->RET_k
# is a faithful, out-of-sample-measurable stand-in for RET_1->target -- covering
# the dominant momentum that the feature->RET_1 proxy above could not test.


# %%
def lag1_ac(df):
    return pd.Series(
        {k: df[f"RET_{k + 1}"].corr(df[f"RET_{k}"], method="spearman") for k in range(1, 20)}
    )


real_boundary = train["RET_1"].corr(
    target, method="spearman"
)  # RET_1 -> target (the real signal)
ac_tr = lag1_ac(train)
ac_te = lag1_ac(test)

print(f"RET_1 -> target (train, the real signal): {real_boundary:+.4f}")
print(f"RET_2 -> RET_1  (train, shifted analog):  {ac_tr[1]:+.4f}")
print(f"RET_2 -> RET_1  (test,  shifted analog):  {ac_te[1]:+.4f}")
print(f"\nlag-1 autocorr mean  train={ac_tr.mean():+.4f}  test={ac_te.mean():+.4f}")
print(
    f"corr(train, test) across lags: {ac_tr.corr(ac_te):.3f}   "
    f"sign agree: {int((np.sign(ac_tr) == np.sign(ac_te)).sum())}/{len(ac_tr)}"
)
print(
    pd.DataFrame({"RET(k+1)->RET(k)_train": ac_tr, "test": ac_te}).to_string(
        float_format=lambda x: f"{x:+.4f}"
    )
)
