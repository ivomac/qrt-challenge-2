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
# # Data Overview
#
# Survey the data structure, distributions, NaN patterns, and train/test alignment.
#
# **Data structure:**
# - SVOL_1 has 73.5% NaN rate; NaN presence is ALLOC-dependent (24 allocs 0% NaN,
#   254 allocs >50% NaN). LightGBM handles NaN natively, no imputation needed.
# - ALLOC 14 and 46 have only 19 train rows each (ALLOC 46 also has full-feature
#   duplicated rows). Both are dropped before modeling.
# - 2,522 unique TS, 278 unique ALLOC, 4 GROUPs. (TS, ALLOC) pairs are unique.
# - RET_0 (target) is 100% NaN in test by design.
#
# **Feature distributions:**
# - RET_* features follow approximate non-central t-distribution. 1% tail-trimming
#   (clipping to +/-0.01) is sufficient.
# - SVOL_* features are bimodal with peaks at +/-1 and heavy tails (kurtosis 359).
# - TURN has 3-5 log-normal modes spanning 10^-12 to 10.
#
# **train vs test:**
# - No timestamp overlap. All 278 ALLOCs appear in both sets. Train row counts per
#   TS vary widely (19-276); test is uniform (102-116 per ALLOC).
# - GROUP distribution shifts: GROUP 3 underrepresented in test (-7 pp), GROUP 1
#   overrepresented (+6 pp).
# - KS tests find statistically significant train-test shifts in 35/41 features,
#   but the shifts are practically small: mean |D_mean_pp| = 0.026 (2.6% of a
#   standard deviation) after clipping. Clipping does not reduce KS because the
#   shift is in the bulk, not the tails.
# - Per-ALLOC SNR train-test Pearson correlation rises monotonically from 0.21
#   (RET_1) to 0.73 (RET_20): longer-horizon cross-sectional signal is more
#   persistent.
# - Per-allocation target bias varies substantially (shrunk deviations from -3.3% to
#   +12.5%) but is unstable: in-sample Spearman rho=+0.74, out-of-sample drops to
#   rho=+0.28. ALLOC aggregates will be effective in train/val but might not carry
#   over in test.

# %%
import tools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot_2samples

sns.set_theme()

# %%
train, test = tools.load("1a-preprocessed/full")

base_float_cols = [
    c
    for c in train.columns
    if c not in {"TS", "ALLOC", "GROUP", "ROW_ID"} and train[c].dtype.kind == "f"
]

# %% [markdown]
# # NaNs

# %%
for name, df in [("train", train), ("test", test)]:
    print(f"NaNs {name:8s}: {df.isna().sum().sum()}")

# %%
nan_counts = train.isna().sum()
nan_cols = nan_counts[nan_counts > 0] / len(train)
print(f"Columns with nan: {len(nan_cols)}")
print(nan_cols.to_string(float_format=lambda x: f"{x:.2%}"))

# %%
# NaN per allocation for SVOL_1
nan_rate_by_alloc = train.groupby("ALLOC", observed=False)["SVOL_1"].apply(
    lambda x: x.isna().mean()
)
print(f"Unique NaN rates for SVOL_1: {nan_rate_by_alloc.nunique()}")
print(f"Allocations with 0% NaN: {(nan_rate_by_alloc == 0).sum()}")
print(f"Allocations with >50% NaN: {(nan_rate_by_alloc > 0.5).sum()}")
print("NaN rate by GROUP (SVOL_1):")
print(
    train.groupby("GROUP", observed=False)["SVOL_1"]
    .apply(lambda x: x.isna().mean() * 100)
    .to_string()
)

# %%
train_nan = train.isna().sum()
test_nan = test.isna().sum()
n_train = len(train)
n_test = len(test)

nan_df = pd.DataFrame(
    {
        "train_n": train_nan,
        "train_%": (train_nan / n_train * 100).round(2),
        "test_n": test_nan,
        "test_%": (test_nan / n_test * 100).round(2),
        "D_pp": (test_nan / n_test - train_nan / n_train) * 100,
    }
)
nan_df = nan_df[(nan_df["train_n"] > 0) | (nan_df["test_n"] > 0)]
nan_df["abs_D"] = nan_df["D_pp"].abs()
nan_df = nan_df.sort_values("abs_D", ascending=False)

print(f"Features with any NaN: {len(nan_df)}")
print(f"  |D| > 1pp: {(nan_df['abs_D'] > 1).sum()}")
print(f"  |D| > 0.5pp: {(nan_df['abs_D'] > 0.5).sum()}")
print()
print(nan_df.drop(columns=["abs_D"]).to_string(float_format=lambda x: f"{x:.2f}"))

# %%
shifted_nan = nan_df.index.tolist()
print("-- Features with NaNs --")
for feat in shifted_nan:
    for name, df in [("train", train), ("test", test)]:
        per_ts = df.loc[df[feat].isna()].groupby("TS", observed=False).size()
        if len(per_ts) == 0:
            print(f"  {feat:8s} {name}: 0 TS with NaN")
        else:
            print(
                f"  {feat:8s} {name}: {len(per_ts):>4} TS with NaN, "
                f"{int(per_ts.min()):>4}-{int(per_ts.max()):<4} rows/TS"
            )
    print()

# %% [markdown]
# # Returns & Volumes

# %%
for prefix, color, extreme in [("RET_", "steelblue", 0.01), ("SVOL_", "darkorange", 10)]:
    cols = [c for c in train.columns if c.startswith(prefix)]
    all_vals = pd.concat([train[c].dropna() for c in cols])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_vals, bins=500, color=color, alpha=0.7, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel(f"{prefix} value")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"All {prefix} values (log y-axis)")
    plt.show(block=False)


# %%
cols = [c for c in train.columns if c.startswith("RET_")]

combined = train[cols].values.ravel()
combined = pd.Series(combined).dropna()

print("\nRET stats")
print(combined.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
print(f"\nskew = {combined.skew()}")
print(f"kurt = {combined.kurt()}\n")

for thresh in np.logspace(-2, -1, 10):
    combined_drop = combined[(combined < thresh) & (combined > -thresh)]
    out_frac = 1 - len(combined_drop) / len(combined)
    print(
        f"within <{thresh:.2e}: out_frac={out_frac:.1e} skew={combined_drop.skew():.2f}  kurt={combined_drop.kurt():.2f}"
    )

# %%
thresh = 5e-2

combined_drop = combined[(combined < thresh) & (combined > -thresh)].dropna()

nu = 4.1
nc = 0.04
mu = -0.0001
sigma = 0.00215

XLIM = (-thresh, thresh)
bins = np.linspace(*XLIM, 201)
x = np.linspace(*XLIM, 500)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

ax1.hist(combined_drop, bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="none")
ax1.set_xlim(XLIM)
ax1.set_xlabel("Return")
ax1.set_ylabel("Density")
ax1.set_title("All RET_* stacked")

ax1.plot(x, stats.nct.pdf(x, nu, nc, mu, sigma), color="crimson", linewidth=1.5)

ax1.set_ylim(bottom=1e-5, top=1e3)
ax1.set_yscale("log")

plt.show(block=False)

# %%
cols = [c for c in train.columns if c.startswith("SVOL_")]

combined = train[cols].values.ravel()
combined = pd.Series(combined).dropna()

print(f"\n{prefix} stats")
print(combined.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
print(f"\nskew = {combined.skew()}")
print(f"kurt = {combined.kurt()}\n")

for thresh in np.logspace(1, 2, 10):
    combined_drop = combined[(combined < thresh) & (combined > -thresh)]
    out_frac = 1 - len(combined_drop) / len(combined)
    print(
        f"within <{thresh:.2e}: out_frac={out_frac:.1e} skew={combined_drop.skew():.2f}  kurt={combined_drop.kurt():.2f}"
    )

# %%
thresh = 30

combined_drop = combined[(combined < thresh) & (combined > -thresh)].dropna()

XLIM = (-thresh, thresh)
bins = np.linspace(*XLIM, 401)
x = np.linspace(*XLIM, 500)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

ax1.hist(combined_drop, bins=bins, density=True, alpha=0.7, color="darkorange", edgecolor="none")
ax1.set_xlim(XLIM)
ax1.set_xlabel("SVOL")
ax1.set_ylabel("Density")
ax1.set_title("All VOL stacked")

n_num = len(combined[combined == -1])
p_num = len(combined[combined == 1])
rat = n_num / (n_num + p_num)

n_params = [2.2, -0.33, -0.76, 0.58]
p_params = [2.4, 0.19, 0.89, 0.46]

pdf = rat * stats.nct.pdf(x, *n_params) + (1 - rat) * stats.nct.pdf(x, *p_params)

ax1.plot(x, pdf, color="blue", linewidth=1.5)

ax1.set_ylim(bottom=1e-7, top=1e1)
ax1.set_yscale("log")

plt.show(block=False)

# %% [markdown]
# # Turnover

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

XLIM = (-12, 2)
bins = np.linspace(*XLIM, 201)

for name, turn, ax in [("Train", train["TURN"], ax1), ("Test", test["TURN"], ax2)]:
    ax.hist(np.log(turn).clip(*XLIM), bins=bins, alpha=0.7, color="steelblue", edgecolor="none")
    ax.set_xlim(XLIM)
    ax.set_title(name)
    ax.set_xlabel("log(Turnover)")
    ax.set_ylabel("Frequency")
    ax.axvline(np.log(turn.mean()), color="red", ls="--", label=f"Mean: {turn.mean():.3f}")
    ax.legend()

# %% [markdown]
# # ALLOC / GROUP

# %%
print(f"TS unique timestamps: {train['TS'].nunique()}")
print(f"ALLOC unique:     {train['ALLOC'].nunique()}")
print(f"GROUP unique:          {train['GROUP'].nunique()}")
print("\nALLOCS per GROUP:")
print(train[["ALLOC", "GROUP"]].drop_duplicates()["GROUP"].value_counts().sort_index().to_string())
print(f"\nAvg obs per allocation: {len(train) / train['ALLOC'].nunique():.0f}")
print(f"Avg obs per date:       {len(train) / train['TS'].nunique():.0f}")

# %%
# (TS, ALLOC) are unique:
sum(train[["TS", "ALLOC"]].groupby(by=["TS", "ALLOC"], observed=True).value_counts() > 1)


# %%
def fit_prior_strength(n_arr: np.ndarray, k_arr: np.ndarray) -> float:
    # method-of-moments Beta-Binomial concentration: pseudo-obs to shrink toward
    p = k_arr / n_arr
    m = np.average(p, weights=n_arr)
    var = np.average((p - m) ** 2, weights=n_arr)
    samp = m * (1 - m) * np.mean(1 / n_arr)  # variance expected from sampling alone
    excess = var - samp
    if excess <= 0:
        return 1e6  # no real dispersion -> shrink everything to the global mean
    return m * (1 - m) / excess - 1


def balance_stats(by: str) -> pd.DataFrame:
    groups = sorted(train[by].unique())
    out = {}
    for name, series, key_df in [
        ("train", train["RET_1"], train),
        ("target", train["RET_0"], train),
        ("test", test["RET_1"], test),
    ]:
        keys = key_df[by].values
        pos = series.values > 0
        valid = ~np.isnan(series.values)
        n_arr = np.array([((keys == g) & valid).sum() for g in groups], dtype=float)
        k_arr = np.array([(pos & (keys == g) & valid).sum() for g in groups], dtype=float)
        p0 = np.average(k_arr / np.maximum(n_arr, 1), weights=n_arr)
        strength = fit_prior_strength(n_arr[n_arr > 0], k_arr[n_arr > 0])
        a, b = p0 * strength, (1 - p0) * strength
        dev = k_arr / np.maximum(n_arr, 1) - p0
        out[(name, "dev")] = pd.Series(dev, index=groups)
        if name == "train":
            print(f"{by}: EB prior strength={strength:.0f} pseudo-obs, global pos={p0:.4f}")
    df = pd.DataFrame(out)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# %% [markdown]
# Per GROUP and ALLOC: estimate the directional bias (+ = more positive returns than
# the global average) for historical returns (train RET_1), the target (RET_0), and
# future returns (test RET_1). Raw deviations ("dev") are noisy for small groups. The
# Spearman rho measures whether per-alloc target bias persists from train to test.

# %%
for col in ("GROUP", "ALLOC"):
    print(f"\n--- {col} ---")
    bstats = balance_stats(col)
    show = (bstats * 100).round(2)
    print(show.sort_values(("target", "dev")).to_string())

    if col == "ALLOC":
        # does per-alloc directional bias persist across history/target/test?
        print("\nStability (Spearman across allocations):")
        for la, ka, lb, kb in [
            ("train RET_1 dev", ("train", "dev"), "target dev", ("target", "dev")),
            ("train RET_1 dev", ("train", "dev"), "test RET_1 dev", ("test", "dev")),
            ("test RET_1 dev", ("test", "dev"), "target dev", ("target", "dev")),
        ]:
            r, p = stats.spearmanr(bstats[ka], bstats[kb])
            print(f"  {la:20s} vs {lb:20s}: rho={r:+.3f} (p={p:.1e})")

# %%
print(
    train[train.duplicated(subset=[f"RET_{i}" for i in range(1, 11)])][["TS", "ALLOC"]].to_string()
)

# %% [markdown]
# ## Timestamps

# %%
# No overlapping times between train and test
print(set(train["TS"]) & set(test["TS"]))

# %%
train_ts = sorted(train["TS"].unique())
test_ts = sorted(test["TS"].unique())

print(f"Train TS: {train_ts[0]} to {train_ts[-1]} ({len(train_ts)} unique)")
print(f"Test TS:  {test_ts[0]} to {test_ts[-1]} ({len(test_ts)} unique)")
print(f"Test TS first few: {test_ts[:5]}")
print(f"Test TS last few:  {test_ts[-5:]}")

# %%
train_rows_per_ts = train.groupby("TS", observed=False).size()
test_rows_per_ts = test.groupby("TS", observed=False).size()

print("Train rows per TS:")
print(
    f"  min={train_rows_per_ts.min()}, median={train_rows_per_ts.median():.0f}, mean={train_rows_per_ts.mean():.1f}, max={train_rows_per_ts.max()}"
)
print(f"  std={train_rows_per_ts.std():.1f}")
print()
print("Test rows per TS:")
print(
    f"  min={test_rows_per_ts.min()}, median={test_rows_per_ts.median():.0f}, mean={test_rows_per_ts.mean():.1f}, max={test_rows_per_ts.max()}"
)
print(f"  std={test_rows_per_ts.std():.1f}")

# %% [markdown]
# ## Allocations

# %%
train_allocs = set(train["ALLOC"])
test_allocs = set(test["ALLOC"])

only_test = test_allocs - train_allocs
only_train = train_allocs - test_allocs

print(f"Train allocations: {len(train_allocs)}")
print(f"Test allocations:  {len(test_allocs)}")
print(f"In test but not train: {len(only_test)}  {sorted(only_test)[:10]}")
print(f"In train but not test: {len(only_train)}  {sorted(only_train)[:10]}")

# %%
train_rpa = train.groupby("ALLOC", observed=False).size()
test_rpa = test.groupby("ALLOC", observed=False).size()

alloc_df = pd.DataFrame({"train": train_rpa, "test": test_rpa}).fillna(0).astype(int)
alloc_df["test/train"] = alloc_df["test"] / alloc_df["train"]

for split in ["train", "test"]:
    df = alloc_df[split]
    print(split)
    print(df.describe().round(1).to_string())

    outliers = df[df < df.quantile(0.25) * 0.5]
    if len(outliers):
        print(f"Allocations with very few rows ({len(outliers)}):")
        print(outliers.to_string())
        print()

# %% [markdown]
# ## GROUPs

# %%
train_group_pct = train["GROUP"].value_counts(normalize=True).sort_index()
test_group_pct = test["GROUP"].value_counts(normalize=True).sort_index()

print("GROUP distribution:")
print(f"{'GROUP':>6} {'Train':>8} {'Test':>8} {'Diff':>8}")
print("-" * 32)
for g in sorted(set(train_group_pct.index) | set(test_group_pct.index)):
    tp = train_group_pct.get(g, 0) * 100
    ttp = test_group_pct.get(g, 0) * 100
    print(f"{g:>6} {tp:>7.2f}% {ttp:>7.2f}% {ttp - tp:>+7.2f}%")
print()
print("Test rows per GROUP (raw count):")
print(test["GROUP"].value_counts().sort_index().to_string())

# %% [markdown]
# ## Feature Distributions

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, data, label in zip(axes, [train["RET_1"], test["RET_1"]], ["Train", "Test"]):
    ax.hist(data.dropna(), bins=200, density=True, alpha=0.7, label=label)
    ax.set_xlim(-0.01, 0.01)
    ax.set_title(f"RET_1 - {label}")
    ax.set_ylabel("Density")

plt.show(block=False)

print(f"Train RET_1: mean={train['RET_1'].mean():.6f}, std={train['RET_1'].std():.6f}")
print(f"Test  RET_1: mean={test['RET_1'].mean():.6f}, std={test['RET_1'].std():.6f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, data, label in zip(axes, [train["TURN"], test["TURN"]], ["Train", "Test"]):
    ax.hist(np.log(data.dropna()).clip(-12, 10), bins=200, density=True, alpha=0.7, label=label)
    ax.set_title(f"log(TURN): {label}")
    ax.set_ylabel("Density")

plt.show(block=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, data, label in zip(axes, [train["SVOL_1"], test["SVOL_1"]], ["Train", "Test"]):
    valid = data.dropna()
    ax.hist(valid, bins=200, density=True, alpha=0.7, label=label)
    ax.set_xlim(-5, 5)
    ax.set_title(f"SVOL_1 - {label} (non-NaN)")
    ax.set_ylabel("Density")

plt.show(block=False)

for name, data in [("Train", train), ("Test", test)]:
    sv1 = data["SVOL_1"]
    print(
        f"{name}: {sv1.isna().sum() / len(sv1) * 100:.1f}% NaN, mean={sv1.mean():.4f}, std={sv1.std():.4f}"
    )

# %% [markdown]
# ## ROW_ID

# %%
train_rid = train.index
test_rid = test.index

print(f"Train ROW_ID range: {train_rid.min()} to {train_rid.max()}")
print(f"Test  ROW_ID range: {test_rid.min()} to {test_rid.max()}")
print(f"Train ROW_ID nunique: {train_rid.nunique()}")
print(f"Test  ROW_ID nunique: {test_rid.nunique()}")

gap = test_rid.min() - train_rid.max()
print(f"Gap between train max and test min: {gap}")
if gap <= 0:
    overlap = set(train_rid) & set(test_rid)
    print(f"Overlapping ROW_IDs: {len(overlap)}")

# %% [markdown]
# ## Sparse Allocations

# %%
rows_per_alloc = train.groupby("ALLOC", observed=False).size()
sparse = rows_per_alloc[rows_per_alloc < 50]
print(f"Sparse allocations (<50 train rows): {len(sparse)} / 278")
if len(sparse) > 0:
    print()
    print("Bottom 10 by row count:")
    print(rows_per_alloc.sort_values().head(10).to_string())

# %% [markdown]
# ## TS Ramp-up

# %%
ts_order = sorted(train["TS"].unique())
rows_per_ts = train.groupby("TS", observed=False).size().reindex(ts_order)
print(f"First 10 TS rows: {rows_per_ts.head(10).to_string()}")

# %% [markdown]
# ## RET_1 Correlation Over Time

# %%
ts_values = sorted(train["TS"].unique())
recent_ts = set(ts_values[-120:])
early_ts = set(ts_values[:120])

for label, ts_set in [("First 120 train TS", early_ts), ("Last 120 train TS", recent_ts)]:
    mask = train["TS"].isin(ts_set)
    r = train.loc[mask, "RET_1"].corr(train.loc[mask, "RET_0"])
    n = mask.sum()
    print(f"{label}: RET_1-target corr={r:.4f}  (n={n})")

# %% [markdown]
# ## Density difference plots
#
# For each feature: difference in normalized density (test - train).
# Blue = test has more mass in that bin, red = train has more.

# %%
n_feats = len(base_float_cols)
n_cols = 3
n_rows = (n_feats + n_cols - 1) // n_cols
print(f"Base float features: {n_feats}")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 2.5))
axes = axes.flatten()

for i, feat in enumerate(base_float_cols):
    ax = axes[i]
    train_vals = train[feat].dropna().values
    test_vals = test[feat].dropna().values

    if len(train_vals) == 0 or len(test_vals) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(feat, fontsize=8)
        continue

    lo = np.percentile(np.concatenate([train_vals, test_vals]), 0.5)
    hi = np.percentile(np.concatenate([train_vals, test_vals]), 99.5)
    bins = np.linspace(lo, hi, 60)

    h_train, _ = np.histogram(train_vals, bins=bins, density=True)
    h_test, _ = np.histogram(test_vals, bins=bins, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    diff = h_test - h_train
    width = bins[1] - bins[0]

    colors = ["#d62728" if d < 0 else "#1f77b4" for d in diff]
    ax.bar(centers, diff, width=width, color=colors, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(feat, fontsize=8)
    ax.tick_params(labelsize=6)

for j in range(n_feats, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Density difference (test - train)", fontsize=12)

plt.show(block=False)

# %% [markdown]
# # Distribution moment shifts

# %%
n_boot = 200
rng = np.random.default_rng(42)

RET_COLS = [f"RET_{i}" for i in range(1, 21)]
SVOL_COLS = [f"SVOL_{i}" for i in range(1, 21)]

for family, cols in [("RET", RET_COLS), ("SVOL", SVOL_COLS), ("TURN", ["TURN"])]:
    train_vals = train[cols].values.ravel()
    train_vals = train_vals[~np.isnan(train_vals)]
    test_vals = test[cols].values.ravel()
    test_vals = test_vals[~np.isnan(test_vals)]

    n_sample = len(test_vals)
    boot = np.array([rng.choice(train_vals, size=n_sample, replace=False) for _ in range(n_boot)])

    def ci(stat_fn, observed):
        boot_stats = [stat_fn(boot[i]) for i in range(n_boot)]
        lo, hi = np.percentile(boot_stats, [2.5, 97.5])
        outside = "*" if observed < lo or observed > hi else " "
        return lo, hi, outside

    std_obs = np.std(test_vals) / np.std(train_vals)
    kurt_obs = stats.kurtosis(test_vals)
    skew_obs = stats.skew(test_vals)
    mad_train = np.median(np.abs(train_vals - np.median(train_vals)))
    mad_test = np.median(np.abs(test_vals - np.median(test_vals)))
    mad_obs = mad_test / mad_train if mad_train > 0 else np.nan

    std_lo, std_hi, std_flag = ci(lambda x: np.std(x) / np.std(train_vals), std_obs)
    kurt_lo, kurt_hi, kurt_flag = ci(stats.kurtosis, kurt_obs)
    skew_lo, skew_hi, skew_flag = ci(stats.skew, skew_obs)
    mad_lo, mad_hi, mad_flag = ci(
        lambda x: np.median(np.abs(x - np.median(x))) / mad_train,
        mad_test,
    )

    print(f"{family}  (n_train={len(train_vals):,}  n_test={n_sample:,})")
    print(f"  {'':10s} {'train':>10} {'test':>10} {'CI_lo':>10} {'CI_hi':>10}")
    print(
        f"  {'std_ratio':10s} {1.0:>10.4f} {std_obs:>10.4f} {std_lo:>10.4f} {std_hi:>10.4f} {std_flag}"
    )
    print(
        f"  {'MAD_ratio':10s} {1.0:>10.4f} {mad_obs:>10.4f} {mad_lo:>10.4f} {mad_hi:>10.4f} {mad_flag}"
    )
    print(
        f"  {'kurtosis':10s} {stats.kurtosis(train_vals):>10.3f} {kurt_obs:>10.3f} {kurt_lo:>10.3f} {kurt_hi:>10.3f} {kurt_flag}"
    )
    print(
        f"  {'skew':10s} {stats.skew(train_vals):>10.3f} {skew_obs:>10.3f} {skew_lo:>10.3f} {skew_hi:>10.3f} {skew_flag}"
    )
    print()

# %% [markdown]
# ## Per-feature KS tests
#
# Two-sample Kolmogorov-Smirnov test per base feature: H0 = train and test
# come from the same distribution. Bonferroni correction for 41 tests.

# %%
from scipy.stats import ks_2samp  # noqa: E402

ks_results = []
for feat in base_float_cols:
    tv = train[feat].dropna().values
    sv = test[feat].dropna().values
    if len(tv) < 100 or len(sv) < 100:
        continue
    stat, pval = ks_2samp(tv, sv)
    ks_results.append({"feature": feat, "KS_stat": stat, "p_value": pval})

ks_df = pd.DataFrame(ks_results).sort_values("KS_stat", ascending=False)
ks_df["p_bonf"] = np.minimum(ks_df["p_value"] * len(ks_df), 1.0)
ks_df["significant"] = ks_df["p_bonf"] < 0.01
print(
    f"Features with significant shift (Bonferroni p<0.01): {ks_df['significant'].sum()} / {len(ks_df)}"
)
print()
print(ks_df.to_string(float_format=lambda x: f"{x:.4f}"))

# %% [markdown]
# ## KS vs clip level
#
# How much does clipping reduce the train-test distribution shift?
# For SVOL features: clip at ±3, ±5, ±7.5, ±10, ±20, no clip.
# For RET features:  clip at ±0.002, ±0.005, ±0.01, ±0.02, ±0.05, no clip.

# %%
SVOL_CLIP_LEVELS = [3.0, 5.0, 7.5, 10.0, 20.0, np.inf]
RET_CLIP_LEVELS = [0.002, 0.005, 0.01, 0.02, 0.05, np.inf]

RET_COLS_KS = [f"RET_{i}" for i in range(1, 21)]
SVOL_COLS_KS = [f"SVOL_{i}" for i in range(1, 21)]


def ks_at_clip(family_cols, clip_val):
    stats_list = []
    for col in family_cols:
        tv = train[col].dropna().values
        sv = test[col].dropna().values
        if len(tv) < 100 or len(sv) < 100:
            continue
        if np.isfinite(clip_val):
            tv = tv.clip(-clip_val, clip_val)
            sv = sv.clip(-clip_val, clip_val)
        stat, _ = ks_2samp(tv, sv)
        stats_list.append(stat)
    return np.mean(stats_list) if stats_list else np.nan


ret_ks = [ks_at_clip(RET_COLS_KS, c) for c in RET_CLIP_LEVELS]
svol_ks = [ks_at_clip(SVOL_COLS_KS, c) for c in SVOL_CLIP_LEVELS]

ret_labels = [f"{c:.3f}" if np.isfinite(c) else "none" for c in RET_CLIP_LEVELS]
svol_labels = [f"{c:.1f}" if np.isfinite(c) else "none" for c in SVOL_CLIP_LEVELS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(ret_labels, ret_ks, "o-", color="steelblue", markersize=6)
ax1.set_xlabel("Clip bound")
ax1.set_ylabel("Mean KS statistic")
ax1.set_title("RET features: KS vs clip level")
ax1.axhline(0.01, color="gray", ls="--", alpha=0.5)
ax1.tick_params(axis="x", rotation=30)

ax2.plot(svol_labels, svol_ks, "o-", color="darkorange", markersize=6)
ax2.set_xlabel("Clip bound")
ax2.set_ylabel("Mean KS statistic")
ax2.set_title("SVOL features: KS vs clip level")
ax2.axhline(0.01, color="gray", ls="--", alpha=0.5)
ax2.tick_params(axis="x", rotation=30)

plt.suptitle("Train-test distribution shift vs clipping aggressiveness", fontsize=12)
plt.show(block=False)

print("\n=== RET: mean KS per clip level ===")
for label, ks in zip(ret_labels, ret_ks):
    print(f"  clip ±{label:>5s}: KS={ks:.4f}")

print("\n=== SVOL: mean KS per clip level ===")
for label, ks in zip(svol_labels, svol_ks):
    print(f"  clip ±{label:>5s}: KS={ks:.4f}")

# %% [markdown]
# # Allocation aggregate significance

# %%
RET_COLS = [f"RET_{i}" for i in range(21)]

dfs = []

print("Pearson correlation between train and test of per-ALLOC SNR")

for ret in RET_COLS:
    df_pair = []
    for name, ds in [
        ("train", train),
        ("test", test),
    ]:
        if ret not in ds.columns:
            continue
        alloc = ds["ALLOC"]

        # allocs 14 and 46 have almost no data
        mask = (alloc != 14) & (alloc != 46) & ~ds[ret].isna()
        y_sub = ds[ret][mask]
        alloc_sub = alloc[mask]

        alloc_stats = y_sub.groupby(alloc_sub, observed=True).agg(["mean", "std", "count"])
        snr = alloc_stats["mean"] / alloc_stats["std"] * np.sqrt(alloc_stats["count"])

        df_pair.append(snr)
    if df_pair[1:]:
        corr = df_pair[0].corr(df_pair[1])
        print(f"{ret:6s}: {corr:.2f}")


# %% [markdown]
# ## Per-day target analysis
#
# Target distribution across training days.

# %%
y_copy = train[["RET_0"]].copy()
y_copy["TS"] = train["TS"].values
ts_tgt = y_copy.groupby("TS", observed=False).agg(
    mean=("RET_0", "mean"),
    std=("RET_0", "std"),
    pos=("RET_0", lambda x: (x > 0).mean()),
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(ts_tgt["mean"] * 10000, bins=60, edgecolor="white")
axes[0].set_xlabel("Mean target (bps)")
axes[0].set_ylabel("Days")
axes[0].set_title("Per-day target mean")
axes[1].hist(ts_tgt["std"] * 10000, bins=60, edgecolor="white")
axes[1].set_xlabel("Std target (bps)")
axes[1].set_title("Per-day target std")
axes[2].hist(ts_tgt["pos"] * 100, bins=60, edgecolor="white")
axes[2].set_xlabel("Positive ratio (%)")
axes[2].set_title("Per-day positive ratio")
plt.suptitle("Target statistics across training days", fontsize=13)
plt.show(block=False)

print((ts_tgt * np.array([10000, 10000, 100])).describe().round(2).to_string())

# %% [markdown]
# ## Q-Q plots
#
# Quantile-quantile: train vs test for every base float feature.

# %%


def get_clip(col: str) -> tuple[float, float] | None:
    """Return (lo, hi) clip bounds derived from test, or None to skip."""
    if col.startswith("SVOL_"):
        return -7.5, 7.5
    if col.startswith("TURN"):
        return 1e-4, 1.6
    return -0.01, 0.01


n_feats = len(base_float_cols)
n_cols = 3
n_rows = (n_feats + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
axes = axes.flatten()

for i, feat in enumerate(base_float_cols):
    ax = axes[i]
    tv = train[feat].dropna().values
    sv = test[feat].dropna().values

    if len(tv) == 0 or len(sv) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(feat)
        continue

    clip = get_clip(feat)
    if clip is not None:
        lo, hi = clip
        tv = tv.clip(lo, hi)
        sv = sv.clip(lo, hi)

    qqplot_2samples(tv, sv, xlabel="train", ylabel="test", ax=ax, line="45")
    ax.set_title(feat + ("*" if clip is not None else ""))
    ax.tick_params(labelsize=6)

for j in range(n_feats, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Q-Q plots: Train vs Test (starred = clipped)", fontsize=12)
plt.show(block=False)

# %% [markdown]
# ## Post-clip distribution comparison
#
# Per-feature comparison after clipping at the Q-Q bounds used in 3a
# (RET ±0.01, SVOL ±7.5, TURN [1e-4, 1.6]).
# D_mean_pp = (test_mean - train_mean) / train_std (in per-unit std).
# MAD_ratio = test_MAD / train_MAD (robust scale ratio).

# %%
RET_COLS = [f"RET_{i}" for i in range(1, 21)]
SVOL_COLS = [f"SVOL_{i}" for i in range(1, 21)]

CLIP_BOUNDS = {}
for c in RET_COLS:
    CLIP_BOUNDS[c] = (-0.01, 0.01)
for c in SVOL_COLS:
    CLIP_BOUNDS[c] = (-7.5, 7.5)
CLIP_BOUNDS["TURN"] = (1e-4, 1.6)

rows = []
for feat, (lo, hi) in CLIP_BOUNDS.items():
    if feat not in train.columns:
        continue
    tv = train[feat].dropna().clip(lo, hi)
    sv = test[feat].dropna().clip(lo, hi)
    rows.append(
        {
            "feature": feat,
            ("mean", "train"): tv.mean(),
            ("mean", "test"): sv.mean(),
            ("std", "train"): tv.std(),
            ("std", "test"): sv.std(),
            ("skew", "train"): tv.skew(),
            ("skew", "test"): sv.skew(),
            ("kurt", "train"): tv.kurt(),
            ("kurt", "test"): sv.kurt(),
            ("ratio", "std"): sv.std() / tv.std() if tv.std() > 0 else np.nan,
            ("shift", "D_pp"): (sv.mean() - tv.mean()) / tv.std(),
        }
    )


clip_df = pd.DataFrame(rows, index=[r["feature"] for r in rows]).drop(columns="feature")
clip_df.columns = pd.MultiIndex.from_tuples(clip_df.columns)
print("=== Post-clip distribution comparison ===")
print(clip_df.to_string(float_format=lambda x: f"{x:.1e}"))
print(f"\nMean |D_mean_pp|: {clip_df[('shift', 'D_pp')].abs().mean():.1e}")

# %% [markdown]
# ## Allocation-level aggregates
#
# Within-allocation z-score: for each (ALLOC, feature) pair,
# z = (test_mean - train_mean) / (train_std / sqrt(n_train)).
# Large |z| flags a feature whose per-alloc mean shifted between train and test.
#
# Real shifts: ALLOC 103 and 90 show SVOL flipping from ~+1 to ~-0.2 in test —
# a genuine regime change for those allocs.
# Artifacts: sparse allocs (14, 46, 19 rows each) produce near-zero SEM and
# astronomical z; TURN is the biggest offender. Ignore these.

# %%
train_am = train.groupby("ALLOC", observed=False)[base_float_cols].mean()
test_am = test.groupby("ALLOC", observed=False)[base_float_cols].mean()
common = train_am.index.intersection(test_am.index)

az = pd.DataFrame(
    {
        "train_mean": train_am.loc[common].mean(),
        "test_mean": test_am.loc[common].mean(),
    }
)

# Within-alloc z-score: per (alloc, feature) pair.
# z[a,f] = (test_mean[a,f] - train_mean[a,f]) / (train_std[a,f] / sqrt(train_n[a,f]))
alloc_train_std = train.groupby("ALLOC", observed=False)[base_float_cols].std()
alloc_train_n = train.groupby("ALLOC", observed=False)[base_float_cols].count()
alloc_sem = alloc_train_std / np.sqrt(alloc_train_n)

diff = test_am.loc[common] - train_am.loc[common]
z_within = (diff / alloc_sem.loc[common]).fillna(0)

# Top (alloc, feature) pairs with largest |z|
stacked = z_within.stack().rename("z_within").reset_index()
top = stacked.iloc[np.abs(stacked["z_within"]).values.argsort()[::-1]]
top["train"] = train_am.loc[common].stack().loc[top.set_index(["ALLOC", "level_1"]).index].values
top["test"] = test_am.loc[common].stack().loc[top.set_index(["ALLOC", "level_1"]).index].values
print(top.head(50).to_string(float_format=lambda x: f"{x:.4f}"))
