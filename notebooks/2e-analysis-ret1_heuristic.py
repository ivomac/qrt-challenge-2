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
# # RET_1 Sign Heuristic Baseline
#
# RET_1 is the most correlated feature. The simplest possible model is predict target using RET_1's sign.
#
# - Baseline accuracy from sign(RET_1) is 51.9%.
# - The signal is consistent across days (59% of days beat 50%).
# - The accuracy-per-bin plot shows how the predictive power of RET_1 varies with its magnitude:
#   * Small |RET_1| values have near-random accuracy, even lower than 50% at near-zero.
#   * Larger |RET_1| values are more predictive.
# - Threshold optimization---optimal k in (pred > k) sign prediction---gives almost no accuracy increase.
# - Per-ALLOC optimized thresholds do NOT generalize. Dead end.

# %%
import tools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sns.set_theme()
np.random.seed(42)

# %%
train, test = tools.load("1a-preprocessed/full")

# %%
# Simple baseline: predict sign of RET_1 as the target sign
ret1 = train["RET_1"]
target = train["RET_0"]
valid = ret1.notna()

y_true_sign = (target[valid] > 0).astype(int)
y_pred_sign = (ret1[valid] > 0).astype(int)

acc = accuracy_score(y_true_sign, y_pred_sign)
bal_acc = balanced_accuracy_score(y_true_sign, y_pred_sign)

print("=== RET_1 sign replication baseline ===")
print(f"Balanced accuracy: {bal_acc * 100:.2f}%")
print(
    f"Baseline (predict majority class): {max(y_true_sign.mean(), 1 - y_true_sign.mean()) * 100:.2f}%"
)

# Is the RET_1 signal consistent across days, or driven by a few?
print("\n=== Signal stability: per-TS accuracy of sign(RET_1) ===")
ts_accs = []
for ts in train["TS"].unique():
    mask = train["TS"] == ts
    r1 = ret1[mask]
    tgt = target[mask]
    valid_ts = r1.notna()
    if valid_ts.sum() < 5:
        continue
    acc_ts = accuracy_score((tgt[valid_ts] > 0).astype(int), (r1[valid_ts] > 0).astype(int))
    ts_accs.append(acc_ts)

ts_accs = np.array(ts_accs)
print(f"Mean per-TS accuracy: {ts_accs.mean() * 100:.2f}%")
print(f"Std per-TS accuracy:  {ts_accs.std() * 100:.2f}%")
print(f"% days above 50%:     {(ts_accs > 0.5).mean() * 100:.1f}%")
print(f"% days above 55%:     {(ts_accs > 0.55).mean() * 100:.1f}%")


# %% [markdown]
# The RET_1 signal strat beats random choice on 59% of days.

# %%
# Diagnostic: accuracy by RET_1 magnitude (equal-count bins)
y_true_ret1 = target[valid].values
y_pred_ret1 = ret1[valid].values

bin_data = tools.accuracy_by_bin(y_true_ret1, y_pred_ret1, n_bins=30)
tools.plot_accuracy_by_bin(bin_data, title="Accuracy per RET_1 bin")
plt.show(block=False)
tools.print_accuracy_table(bin_data, y_pred_ret1)

# %% [markdown]
# # Threshold Optimization Strategies
#
# The accuracy-by-bin analysis shows that small |RET_1| values are unreliable (<50% accuracy near zero).
# We explore different strategies to find an optimal decision threshold beyond t=0 in `RET_1 > t`.

# %%
y_true_binary = (y_true_ret1 > 0).astype(int)

# %% [markdown]
# ## Shifted Sign Threshold (Percentile Scan)
#
# Scan percentile-based thresholds and pick the one maximizing balanced accuracy. This lets the decision boundary shift from 0 to compensate for distribution skew or signal asymmetry.

# %%
best_t_shift, best_acc_shift = tools.plot_threshold_optimization(y_true_ret1, y_pred_ret1)

# %% [markdown]
# ## Per-ALLOC Thresholds
#
# Compute a separate optimal threshold (via percentile scan) for each ALLOC group.

# %%
alloc_valid = train["ALLOC"][valid].values

alloc_thresholds = {}
alloc_results = []
for alloc in np.unique(alloc_valid):
    mask = alloc_valid == alloc
    n = mask.sum()
    if n < 50:
        continue
    y_t = y_true_ret1[mask]
    y_p = y_pred_ret1[mask]

    thresholds = np.percentile(y_p, q=np.arange(30, 71, 2))
    accs = [
        balanced_accuracy_score((y_t > 0).astype(int), (y_p > t).astype(int)) for t in thresholds
    ]
    best_idx = np.argmax(accs)
    alloc_thresholds[alloc] = thresholds[best_idx]
    alloc_results.append(
        {
            "ALLOC": alloc,
            "n": n,
            "t_opt": thresholds[best_idx],
            "bal_acc": accs[best_idx] * 100,
        }
    )

alloc_df = pd.DataFrame(alloc_results).set_index("ALLOC").sort_values("t_opt")
print(alloc_df.to_string(float_format=lambda x: f"{x:.2e}" if abs(x) < 0.01 else f"{x:.2f}"))

# Apply per-ALLOC thresholds globally
pred_per_alloc = np.zeros(len(y_pred_ret1), dtype=int)
for alloc, t in alloc_thresholds.items():
    mask = alloc_valid == alloc
    pred_per_alloc[mask] = (y_pred_ret1[mask] > t).astype(int)

global_acc_per_alloc = balanced_accuracy_score(y_true_binary, pred_per_alloc)
print(f"\nGlobal balanced acc with per-ALLOC thresholds: {global_acc_per_alloc * 100:.2f}%")

# Bar chart
_, ax = plt.subplots(figsize=(10, 4))
allocs_sorted = alloc_df.index.tolist()
colors = ["red" if t < 0 else "green" for t in alloc_df["t_opt"]]
ax.bar(range(len(allocs_sorted)), alloc_df["t_opt"], color=colors)
ax.axvline(np.median(allocs_sorted), color="gray", ls="-", linewidth=1)
min_tick = (
    -max(
        0.001,
        np.percentile(np.abs(alloc_df["t_opt"].values[alloc_df["t_opt"] < 0]), 90)
        if (alloc_df["t_opt"] < 0).any()
        else 0.001,
    )
    * 1.5
)
max_tick = (
    max(
        0.001,
        np.percentile(alloc_df["t_opt"].values[alloc_df["t_opt"] > 0], 90)
        if (alloc_df["t_opt"] > 0).any()
        else 0.001,
    )
    * 1.5
)
ax.set_ylim(min(min_tick, -0.0001), max(max_tick, 0.0001))
ax.axhline(0, color="blue", ls="--", linewidth=1)
step = max(1, len(allocs_sorted) // 20)
ax.set_xticks(range(0, len(allocs_sorted), step))
ax.set_xticklabels(
    [allocs_sorted[i] for i in range(0, len(allocs_sorted), step)],
    rotation=45,
    ha="right",
    fontsize=8,
)
ax.set_ylabel("Optimal threshold")
ax.set_title("Optimal RET_1 threshold per ALLOC")
plt.show(block=False)

# %% [markdown]
# ## Per-Day (TS) Thresholds
#
# Market regimes shift across days. Compute optimal thresholds per trading day
# to see how the optimal decision boundary varies.

# %%
ts_valid = train["TS"][valid].values

ts_thresholds = {}
ts_results = []
for ts in np.unique(ts_valid):
    mask = ts_valid == ts
    n = mask.sum()
    if n < 20:
        continue
    y_t = y_true_ret1[mask]
    y_p = y_pred_ret1[mask]

    thresholds = np.percentile(y_p, q=np.arange(30, 71, 2))
    accs = [
        balanced_accuracy_score((y_t > 0).astype(int), (y_p > t).astype(int)) for t in thresholds
    ]
    best_idx = np.argmax(accs)
    ts_thresholds[ts] = thresholds[best_idx]
    ts_results.append(
        {
            "TS": ts,
            "n": n,
            "t_opt": thresholds[best_idx],
            "bal_acc": accs[best_idx] * 100,
        }
    )

ts_df = pd.DataFrame(ts_results).set_index("TS")
t_opts = ts_df["t_opt"].values

print(f"Days processed: {len(t_opts)}")
print(f"Daily threshold mean:   {t_opts.mean():.2e}")
print(f"Daily threshold std:    {t_opts.std():.2e}")
print(f"% days with t_opt < 0:  {(t_opts < 0).mean() * 100:.1f}%")
print(f"% days with t_opt > 0:  {(t_opts > 0).mean() * 100:.1f}%")

# Apply per-day thresholds globally
pred_per_ts = np.zeros(len(y_pred_ret1), dtype=int)
for ts, t in ts_thresholds.items():
    mask = ts_valid == ts
    pred_per_ts[mask] = (y_pred_ret1[mask] > t).astype(int)

global_acc_per_ts = balanced_accuracy_score(y_true_binary, pred_per_ts)
print(f"\nGlobal balanced acc with per-TS thresholds: {global_acc_per_ts * 100:.2f}%")

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.hist(t_opts, bins=40, color="steelblue", edgecolor="white")
ax1.axvline(0, color="red", ls="--", linewidth=1.5, label="default (0)")
ax1.axvline(
    t_opts.mean(), color="green", ls="--", linewidth=1.5, label=f"mean ({t_opts.mean():.2e})"
)
ax1.set_xlabel("Optimal threshold")
ax1.set_ylabel("Number of days")
ax1.set_title("Distribution of per-day optimal thresholds")
ax1.legend()

ax2.scatter(t_opts, ts_df["bal_acc"], alpha=0.5, s=15)
ax2.axhline(51.89, color="gray", ls="--", label="baseline (51.89%)")
ax2.axvline(0, color="red", ls="--", linewidth=1)
ax2.set_xlabel("Optimal threshold")
ax2.set_ylabel("Balanced Accuracy (%)")
ax2.set_title("Daily accuracy vs daily optimal threshold")
ax2.legend()

plt.show(block=False)

# %% [markdown]
# # Threshold stability: 50-50 TS-split cross-validation
#
# The per-TS and per-ALLOC threshold optimization above uses the full dataset
# (in-sample). To check whether optimal thresholds generalize, we split the
# 2522 TS values into two disjoint halves, compute thresholds on each half, and
# cross-apply them to the opposite half.

# %%
ts_unique = sorted(train["TS"].unique())
rng = np.random.default_rng(42)
rng.shuffle(ts_unique)
mid = len(ts_unique) // 2
ts_A = set(ts_unique[:mid])
ts_B = set(ts_unique[mid:])

mask_A = train["TS"].isin(ts_A).values[valid]
mask_B = train["TS"].isin(ts_B).values[valid]

print(f"Split A: {len(ts_A)} TS, {mask_A.sum()} rows")
print(f"Split B: {len(ts_B)} TS, {mask_B.sum()} rows")
print(f"Overlap: {len(ts_A & ts_B)}")


# %%
def compute_split_thresholds(y_true, y_pred, ts_ids, alloc_ids, min_ts=20, min_alloc=50):
    """Compute per-TS and per-ALLOC optimal thresholds from one data split."""
    ts_thresh = {}
    for ts in np.unique(ts_ids):
        m = ts_ids == ts
        if m.sum() < min_ts:
            continue
        yt, yp = y_true[m], y_pred[m]
        thresholds = np.percentile(yp, q=np.arange(30, 71, 2))
        accs = [
            balanced_accuracy_score((yt > 0).astype(int), (yp > t).astype(int)) for t in thresholds
        ]
        ts_thresh[ts] = thresholds[np.argmax(accs)]

    alloc_thresh = {}
    for alloc in np.unique(alloc_ids):
        m = alloc_ids == alloc
        if m.sum() < min_alloc:
            continue
        yt, yp = y_true[m], y_pred[m]
        thresholds = np.percentile(yp, q=np.arange(30, 71, 2))
        accs = [
            balanced_accuracy_score((yt > 0).astype(int), (yp > t).astype(int)) for t in thresholds
        ]
        alloc_thresh[alloc] = thresholds[np.argmax(accs)]

    return ts_thresh, alloc_thresh


def apply_thresholds(y_pred, ts_ids, alloc_ids, ts_thresh, alloc_thresh, y_true):
    """Apply per-TS and per-ALLOC thresholds, return balanced accuracy."""
    pred = np.zeros(len(y_pred), dtype=int)
    for ts, t in ts_thresh.items():
        m = ts_ids == ts
        if m.any():
            pred[m] = (y_pred[m] > t).astype(int)
    for alloc, t in alloc_thresh.items():
        m = (alloc_ids == alloc) & ~np.isin(ts_ids, list(ts_thresh.keys()))
        if m.any():
            pred[m] = (y_pred[m] > t).astype(int)
    # Remaining: fallback to sign
    remaining = ~np.isin(ts_ids, list(ts_thresh.keys())) & ~np.isin(
        alloc_ids, list(alloc_thresh.keys())
    )
    if remaining.any():
        pred[remaining] = (y_pred[remaining] > 0).astype(int)
    return balanced_accuracy_score((y_true > 0).astype(int), pred)


# %%
ts_A_thresh, alloc_A_thresh = compute_split_thresholds(
    y_true_ret1[mask_A],
    y_pred_ret1[mask_A],
    ts_valid[mask_A],
    alloc_valid[mask_A],
)
ts_B_thresh, alloc_B_thresh = compute_split_thresholds(
    y_true_ret1[mask_B],
    y_pred_ret1[mask_B],
    ts_valid[mask_B],
    alloc_valid[mask_B],
)

print(f"Split A: {len(ts_A_thresh)} TS thresholds, {len(alloc_A_thresh)} ALLOC thresholds")
print(f"Split B: {len(ts_B_thresh)} TS thresholds, {len(alloc_B_thresh)} ALLOC thresholds")

# %%
baseline_acc = balanced_accuracy_score(y_true_binary, (y_pred_ret1 > 0).astype(int))

acc_AA = apply_thresholds(
    y_pred_ret1[mask_A],
    ts_valid[mask_A],
    alloc_valid[mask_A],
    ts_A_thresh,
    alloc_A_thresh,
    y_true_ret1[mask_A],
)
acc_BB = apply_thresholds(
    y_pred_ret1[mask_B],
    ts_valid[mask_B],
    alloc_valid[mask_B],
    ts_B_thresh,
    alloc_B_thresh,
    y_true_ret1[mask_B],
)
acc_AB = apply_thresholds(
    y_pred_ret1[mask_B],
    ts_valid[mask_B],
    alloc_valid[mask_B],
    ts_A_thresh,
    alloc_A_thresh,
    y_true_ret1[mask_B],
)
acc_BA = apply_thresholds(
    y_pred_ret1[mask_A],
    ts_valid[mask_A],
    alloc_valid[mask_A],
    ts_B_thresh,
    alloc_B_thresh,
    y_true_ret1[mask_A],
)

print(f"\nBaseline (sign, t=0):                   {baseline_acc * 100:.2f}%")
print(f"In-sample  A->A:                        {acc_AA * 100:.2f}%")
print(f"In-sample  B->B:                        {acc_BB * 100:.2f}%")
print(f"Cross-apply A thresholds to B:          {acc_AB * 100:.2f}%")
print(f"Cross-apply B thresholds to A:          {acc_BA * 100:.2f}%")
print(f"Cross-apply average:                    {(acc_AB + acc_BA) / 2 * 100:.2f}%")
print(
    f"Cross-apply lift over baseline:         {((acc_AB + acc_BA) / 2 - baseline_acc) * 100:.2f}pp"
)

common_alloc = set(alloc_A_thresh.keys()) & set(alloc_B_thresh.keys())
alloc_corr = np.corrcoef(
    [alloc_A_thresh[a] for a in common_alloc],
    [alloc_B_thresh[a] for a in common_alloc],
)[0, 1]
print(f"Threshold correlation across splits:    r={alloc_corr:.3f}")
