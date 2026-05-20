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
# # Signal vs. noise
#
# Financial returns are extremely noisy. Below we decompose its variance into
# day-level, allocation-level, and residual components.
#
# - ~93% of target variance is residual noise after removing day + allocation effects.
# - The day effect (8.3%) dominates the allocation effect (1.8%).
# - The ALLOC variance dominates over the GROUP variance: Within GROUP var >> Across GROUP var.
# - GROUP is weak as predictor, ALLOC favored.
#
# Variance is the wrong currency for a sign metric, so we also translate the
# structure into accuracy ceilings (oracle alloc/day signs, in-sample upper
# bounds), and measure how forecastable the day component is from the
# cross-section of lagged returns.

# %%
import tools

import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

# %%
train, test = tools.load("1a-preprocessed/full")

# %%
target = train["RET_0"].values
var_total = target.var()

ts_means = train.groupby("TS", observed=False)["RET_0"].mean()
alloc_means = train.groupby("ALLOC", observed=False)["RET_0"].mean()

day_var = ts_means.var()
alloc_var = alloc_means.var()

# Residual after removing day effects only
resid_day = train["RET_0"] - train["TS"].map(ts_means)
var_resid_day = resid_day.var()

# Residual after removing both day and allocation effects
resid_both = train["RET_0"] - train["TS"].map(ts_means) - train["ALLOC"].map(alloc_means)
var_resid_both = resid_both.var()

print(f"Total target var: {var_total:.2e}  (std={target.std():.4f})")
print(f"Day effect:       {day_var:.2e}  ({day_var / var_total * 100:.1f}%)")
print(f"Allocation effect:{alloc_var:.2e}  ({alloc_var / var_total * 100:.1f}%)")
print(
    f"Residual (day removed only):     {var_resid_day:.2e}  ({var_resid_day / var_total * 100:.1f}%)"
)
print(
    f"Residual (day + alloc removed):  {var_resid_both:.2e}  ({var_resid_both / var_total * 100:.1f}%)"
)


# %%
t = train["RET_0"]
print("=== TARGET distribution ===")
print(t.describe())
print(f"\nSkewness: {t.skew():.4f}")
print(f"Kurtosis: {t.kurt():.4f}  (Gaussian=3)")
print(f"\nPositive rate: {(t > 0).mean() * 100:.2f}%")

# Tail quantiles
for q in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]:
    print(f"  {q * 100:5.1f}% quantile: {t.quantile(q):.6f}")


# %%
# Day-level metrics
ts_means = train.groupby("TS", observed=False)["RET_0"].mean()

print("=== Day-level statistics ===")
print(f"Number of timestamps: {len(ts_means)}")
print("Day-mean target:")
print(f"  Mean: {ts_means.mean():.3e}")
print(f"  Std:  {ts_means.std():.3e}")
print(f"  Range: [{ts_means.min():.3e}, {ts_means.max():.3e}]")


# %%
alloc_means = (
    train[["ALLOC", "GROUP", "RET_0"]]
    .groupby(["ALLOC", "GROUP"], observed=False)["RET_0"]
    .mean()
    .reset_index()
    .rename(columns={"RET_0": "mean_target"})
)

# Variance between group means vs variance across all allocation means
group_means = alloc_means.groupby("GROUP", observed=False)["mean_target"].mean()
var_between = group_means.var()
var_within = (
    alloc_means.groupby("GROUP", observed=False)["mean_target"].apply(lambda x: x.var()).mean()
)

print(alloc_means.groupby("GROUP", observed=False)["mean_target"].describe().round(6))
print(f"\nVar between groups: {var_between:.2e}")
print(f"Var within groups: {var_within:.2e}")
print(f"Ratio between/within: {var_between / var_within:.2f}")

# %% [markdown]
# ## Accuracy ceilings (oracle decomposition)
#
# Translate the variance components into the accuracy each oracle predictor would
# reach. All use full-sample means, so these are in-sample upper bounds.

# %%
sign_y = train["RET_0"].values > 0
n = len(sign_y)

alloc_mean_series = train.groupby("ALLOC", observed=False)["RET_0"].mean()
ts_mean_row = train["TS"].map(ts_means).values  # day-mean target (oracle)
alloc_mean_row = train["ALLOC"].map(alloc_mean_series).values  # alloc-mean target (oracle)
ret1 = train["RET_1"].values


def acc(pred_pos: np.ndarray) -> float:
    return (pred_pos == sign_y).mean() * 100


print(f"Always-positive:         {acc(np.ones(n, bool)):.2f}%")
print(f"sign(RET_1) heuristic:   {acc(ret1 > 0):.2f}%")
print(f"Oracle alloc-mean sign:  {acc(alloc_mean_row > 0):.2f}%")
print(f"Oracle day-mean sign:    {acc(ts_mean_row > 0):.2f}%")
print(f"Oracle alloc+day sign:   {acc((alloc_mean_row + ts_mean_row) > 0):.2f}%")

# %% [markdown]
# ## Is the day component forecastable from the cross-section?
#
# The day effect is the largest structured chunk but is a contemporaneous common
# shock. On each TS we observe every allocation's lagged returns, so the
# cross-sectional mean of recent returns may forecast the next-day day-mean.

# %%
ret_cols = [f"RET_{i}" for i in range(1, 21)]
day = (
    train[ret_cols]
    .assign(RET_0=train["RET_0"].values, TS=train["TS"].values)
    .groupby("TS", observed=True)
    .mean()
)

corrs = day[ret_cols].corrwith(day["RET_0"]).sort_values()
print("corr(day-mean RET_k, day-mean target) — strongest:")
print(corrs.head(5).to_string(float_format="{:+.3f}".format))
print(corrs.tail(5).to_string(float_format="{:+.3f}".format))

# In-sample R^2 of next-day day-mean from all cross-sectional lag means
A = np.column_stack([day[ret_cols].values, np.ones(len(day))])
b = day["RET_0"].values
coef, *_ = np.linalg.lstsq(A, b, rcond=None)
pred_day = A @ coef
r2 = 1 - ((b - pred_day) ** 2).sum() / ((b - b.mean()) ** 2).sum()
print(f"\nDay-mean target R^2 from cross-sectional lag means: {r2:.3f}")
print(f"Forecastable day variance ~= 8.3% * R^2 = {8.3 * r2:.2f}% of total")

# Per-row accuracy if we predict the forecasted day sign for every alloc that day
pred_day_sign = train["TS"].map(pd.Series(pred_day > 0, index=day.index)).values
print(f"Forecast day-sign accuracy (cross-sectional OLS): {acc(pred_day_sign):.2f}%")
