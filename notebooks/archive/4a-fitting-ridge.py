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
# # Ridge Baseline
#
# Ridge regression as a linear benchmark. How does it compare to the sign(RET_1) baseline of 51.9%?

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

import tools

sns.set_theme()


# %%
N_SPLITS = 12
ALPHA = 2
PREDICT = True

# %%
X_train, X_test, y_train = tools.load("3a-engineering/full")

# %%
include = {
    # "ALLOC",
    # "GROUP",
    # "TURN",
    "RET_1",
    "RET_10",
    "RET_11",
    # "RET_12",
    # "RET_13",
    # "RET_14",
    # "RET_15",
    # "RET_16",
    # "RET_17",
    # "RET_18",
    # "RET_19",
    # "RET_1_cubed",
    # "RET_1_gt_RET_2",
    "RET_1_minus_row_mean_RET_20",
    # "RET_1_minus_row_mean_RET_7",
    # "RET_1_minus_ts_median_RET_1",
    "RET_1_squared",
    # "RET_1_times_RET_2",
    "RET_1_times_allocation_sharpe_row_mean_RET_20",
    "RET_1_times_row_mean_RET_7",
    # "RET_1_times_ts_max_RET_1",
    # "RET_1_times_ts_std_TURN",
    "RET_2",
    # "RET_20",
    "RET_3",
    "RET_4",
    "RET_5",
    "RET_6",
    "RET_7",
    "RET_8",
    "RET_9",
    # "SVOL_1",
    # "SVOL_10",
    # "SVOL_11",
    # "SVOL_12",
    # "SVOL_13",
    # "SVOL_14",
    # "SVOL_15",
    # "SVOL_16",
    # "SVOL_17",
    # "SVOL_18",
    # "SVOL_19",
    # "SVOL_1_div_row_mean_VOLUME_20",
    # "SVOL_1_times_RET_1",
    # "SVOL_2",
    # "SVOL_20",
    # "SVOL_3",
    # "SVOL_4",
    # "SVOL_5",
    # "SVOL_6",
    # "SVOL_7",
    # "SVOL_8",
    # "SVOL_9",
    # "SV_change_1_2",
    # "TS",
    "allocation_max_TURN",
    # "allocation_max_row_mean_RET_20",
    "allocation_mean_TURN",
    "allocation_mean_row_mean_RET_20",
    "allocation_min_TURN",
    # "allocation_sharpe_TURN",
    # "allocation_sharpe_row_mean_RET_20",
    "allocation_std_TURN",
    # "allocation_std_row_mean_RET_20",
    # "div_RET_1_RET_1_squared",
    "div_RET_1_squared_ts_mean_TURN",
    "div_RET_1_squared_ts_mean_row_mean_VOLUME_20",
    # "div_RET_1_squared_ts_std_TURN",
    "div_RET_1_ts_mean_TURN",
    "div_RET_1_ts_mean_row_mean_VOLUME_20",
    # "div_RET_1_ts_std_TURN",
    "div_ts_mean_TURN_ts_mean_row_mean_VOLUME_20",
    "div_ts_mean_TURN_ts_std_TURN",
    "div_ts_std_TURN_ts_mean_row_mean_VOLUME_20",
    # "momentum_5",
    "pos_ret_weighted_5",
    # "row_max_RET_14",
    "row_max_RET_20",
    "row_max_RET_7",
    # "row_max_RET_7_div_row_mean_RET_20",
    # "row_max_SVOL_20",
    # "row_max_SVOL_7",
    # "row_max_VOLUME_20",
    "row_max_VOLUME_7",
    # "row_mean_RET_14",
    # "row_mean_RET_20",
    "row_mean_RET_7",
    # "row_mean_SVOL_20",
    "row_mean_SVOL_7",
    # "row_mean_VOLUME_20",
    # "row_mean_VOLUME_7",
    "row_min_RET_14",
    # "row_min_RET_20",
    "row_min_RET_7",
    # "row_min_SVOL_20",
    "row_min_SVOL_7",
    # "row_min_VOLUME_20",
    # "row_min_VOLUME_7",
    # "row_pos_count_RET_20",
    "row_pos_count_RET_7",
    "row_sharpe_RET_20",
    # "row_sharpe_RET_7",
    # "row_std_RET_14",
    "row_std_RET_20",
    # "row_std_RET_20_times_ts_std_RET_1",
    "row_std_RET_7",
    # "row_std_RET_7_times_ts_std_RET_1",
    "row_std_SVOL_20",
    # "row_std_SVOL_7",
    "row_std_VOLUME_20",
    "row_std_VOLUME_7",
    # "sign_agreement_RET_1_RET_2",
    "ts_demean_TURN",
    # "ts_demean_RET_1",
    "ts_demean_RET_2",
    # "ts_demean_RET_3",
    "ts_demean_RET_4",
    "ts_demean_row_mean_RET_20",
    # "ts_demean_row_mean_VOLUME_20",
    "ts_max_TURN",
    # "ts_max_RET_1",
    # "ts_max_RET_2",
    # "ts_max_RET_3",
    # "ts_max_RET_4",
    "ts_max_row_max_RET_20",
    "ts_mean_TURN",
    # "ts_mean_RET_1",
    # "ts_mean_RET_2",
    # "ts_mean_RET_3",
    "ts_mean_RET_4",
    "ts_mean_row_mean_RET_20",
    "ts_mean_row_mean_RET_7",
    "ts_mean_row_mean_VOLUME_20",
    # "ts_mean_row_std_RET_20",
    "ts_median_RET_1",
    "ts_min_TURN",
    "ts_min_RET_1",
    # "ts_min_RET_2",
    # "ts_min_RET_3",
    "ts_min_RET_4",
    # "ts_min_row_min_RET_20",
    # "ts_rank_TURN",
    # "ts_rank_RET_1",
    # "ts_rank_RET_3",
    "ts_rank_RET_4",
    # "ts_rank_SVOL_1",
    # "ts_rank_row_mean_RET_20",
    "ts_std_TURN",
    "ts_std_RET_1",
    "ts_std_RET_2",
    "ts_std_RET_3",
    "ts_std_RET_4",
    # "vol_ratio",
    # "vol_regime",
    # "vol_ret_agreement_1",
    "vol_weighted_ret_3",
    "zscore_RET_1",
    "zscore_RET_2",
}

feature_cols = [c for c in X_train.columns if c in include]

# %%
predictions = []
scores = []

for fold in range(N_SPLITS):
    X_tr, X_val, y_tr, y_val = tools.load(f"3a-engineering/folds/{fold}")

    X_tr = X_tr[feature_cols].fillna(0.0)
    X_val_pred = X_val[feature_cols].fillna(0.0)

    y_tr = y_tr["target"]
    y_val = y_val["target"]
    ts_val = X_val["TS"].values

    pipeline = make_pipeline(
        StandardScaler(), RidgeCV(alphas=[ALPHA], scoring="neg_mean_squared_error")
    )

    pipeline.fit(X_tr, y_tr)

    model = pipeline.named_steps["ridgecv"]

    train_pred = pipeline.predict(X_tr)
    y_pred = pipeline.predict(X_val_pred)

    predictions.append(
        {
            ("train", "true"): y_tr.values,
            ("train", "pred"): train_pred,
            ("val", "true"): y_val.values,
            ("val", "pred"): y_pred,
            "importances": pd.Series(np.abs(model.coef_), index=feature_cols),
        }
    )

    scores.append(
        {
            "fold": fold + 1,
            **tools.compute_metrics(train=(y_tr.values, train_pred), val=(y_val.values, y_pred)),
        }
    )

    print(f"Fold {fold + 1}: acc={scores[-1][('val', 'acc')] * 100:.2f}%")


# %% [raw]
# Fold 1: acc=53.05%
# Fold 2: acc=51.80%
# Fold 3: acc=52.37%
# Fold 4: acc=52.61%
# Fold 5: acc=52.65%
# Fold 6: acc=52.77%
# Fold 7: acc=52.06%
# Fold 8: acc=52.60%
# Fold 9: acc=53.08%
# Fold 10: acc=52.03%
# Fold 11: acc=52.54%
# Fold 12: acc=52.31%

# %%
print(tools.metrics_table(scores))

# %%
# Threshold optimization

all_y_val = np.concatenate([p[("val", "true")] for p in predictions])
all_y_pred = np.concatenate([p[("val", "pred")] for p in predictions])

best_t, best_acc = tools.plot_threshold_optimization(all_y_val, all_y_pred)


# %%
# accuracy by predicted return magnitude (equal-count bins)

bin_data = tools.accuracy_by_bin(all_y_val, all_y_pred, n_bins=20)

tools.plot_accuracy_by_bin(bin_data)
plt.show()

tools.print_accuracy_table(bin_data, all_y_pred)

# %%
importances = pd.concat([p["importances"].to_frame().T for p in predictions])

tools.format_importances(importances, sort_col="ratio")

# %% [markdown]
# ## Final Model & Test Predictions
#
# Retrain on full training set with the median alpha, predict test set.
#

# %%
if PREDICT:
    tools.clean("4-submissions")
    X_train, X_test, y_train = tools.load("3a-engineering/full")

    ROW_IDS_test = X_test.index

    X_train = X_train[feature_cols].fillna(0.0)
    X_test = X_test[feature_cols].fillna(0.0)

    y_train = y_train["target"]

    pipeline = make_pipeline(
        StandardScaler(), RidgeCV(alphas=[ALPHA], scoring="neg_mean_squared_error")
    )

    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)

    model = pipeline.named_steps["ridgecv"]
    importances = pd.Series(np.abs(model.coef_), index=feature_cols).to_frame().T

    tools.format_importances(importances, sort_col="pct")

    print(f"Test positive rate: {(y_pred_test > 0).mean() * 100:.2f}%")
    print(f"Test prediction range: [{y_pred_test.min():.6f}, {y_pred_test.max():.6f}]")

    sub = pd.DataFrame(
        {
            "ROW_ID": ROW_IDS_test,
            "target": (y_pred_test > 0).astype(int),
        }
    )
    tools.submit("ridge", sub)

# %% [markdown]
# ## Summary
#
# **Ridge results**:
# - CV sign accuracy: **52.36%** (vs RET_1 baseline 51.89%) — +0.47pp edge
# - Days above 50%: **62.5%** (vs RET_1 59.0%) — +3.5pp
# - Mean per-TS rank IC: **+0.0787** (vs RET_1 +0.062)
# - Best alpha consistently near **0.001** (weak regularization — model needs low bias)
#
# **Interpretation**: Ridge edges the simple RET_1 momentum rule by +0.47pp.
# The gain is modest but consistent across folds (low variance). Engineered features
# contribute some linear signal beyond RET_1 alone.
#
# **Bottom line**: Ridge is a valid benchmark with a small but real edge over RET_1.
# LightGBM needs to do better by exploiting non-linear interactions.
#
