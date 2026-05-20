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
# # LightGBM — Binary Classification
#
# Previous models (Ridge, LightGBM MSE) all optimize L2 on raw returns,
# but we evaluate on sign accuracy. This notebook switches to direct
# log-loss optimization with `objective='binary'` and `metric='auc'`.
#
# **Hypothesis**: Aligning loss with evaluation metric should improve
# sign accuracy. AUC directly measures ranking quality (does allocation A
# score higher than allocation B on a given day?), which is what we need.

# %%
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tools

sns.set_theme()

# %%
N_SPLITS = 8
PREDICT = True

# %%
X_train, X_test, y_train = tools.load("3a-engineering/full")

# Convert target to binary
y_train_bin = (y_train["target"].values > 0).astype(int)
pos_rate = y_train_bin.mean()
print(f"Positive rate: {pos_rate:.4f}")

# %%
all_cols = list(X_train.columns)

exclude = {
    "TS",
    "ALLOC",
    "GROUP",
    *(f"SVOL_{i}" for i in range(2, 21)),
    *(f"RET_{k}" for k in range(5, 21)),
    "TURN",
    "RET_1_squared",
    "RET_1_times_RET_2",
    "RET_1_times_row_mean_RET_7",
    "SVOL_1_div_row_mean_VOLUME_20",
    "SVOL_1_times_RET_1",
    "SV_change_1_2",
    "allocation_max_TURN",
    "allocation_min_TURN",
    "allocation_sharpe_TURN",
    "allocation_std_TURN",
    "allocation_std_row_mean_RET_20",
    "div_RET_1_RET_1_squared",
    "div_RET_1_squared_ts_mean_TURN",
    "div_RET_1_squared_ts_mean_row_mean_VOLUME_20",
    "div_RET_1_squared_ts_std_TURN",
    "momentum_5",
    "pos_ret_weighted_5",
    "row_max_RET_20",
    "row_max_RET_7",
    "row_max_RET_7_div_row_mean_RET_20",
    "row_mean_RET_20",
    "row_mean_RET_7",
    "row_mean_VOLUME_20",
    "row_min_RET_20",
    "row_min_RET_7",
    "row_pos_count_RET_20",
    "row_pos_count_RET_7",
    "row_sharpe_RET_7",
    "row_std_RET_20",
    "row_std_RET_20_times_ts_std_RET_1",
    "row_std_RET_7",
    "row_std_RET_7_times_ts_std_RET_1",
    "row_std_VOLUME_20",
    "sign_agreement_RET_1_RET_2",
    "ts_demean_RET_1",
    "ts_demean_row_mean_RET_20",
    "ts_demean_row_mean_VOLUME_20",
    "ts_max_row_max_RET_20",
    "tsrank_TURN",
    "tsrank_RET_1",
    "tsrank_RET_3",
    "tsrank_RET_4",
    "tsrank_SVOL_1",
    "tsrank_row_mean_RET_20",
    "vol_ratio",
    "vol_regime",
    "vol_ret_agreement_1",
    "zscore_RET_1",
    "zscore_RET_2",
}
include = {}

feature_cols = [c for c in X_train.columns if c in include and c not in exclude]
cat_cols = [c for c in feature_cols if isinstance(X_train[c].dtype, pd.CategoricalDtype)]

# %%
lgbm_params = {
    "num_threads": 12,
    "verbose": -1,
    "seed": 42,
    "boost_from_average": True,
    "max_cat_to_onehot": 5,
    "objective": "binary",
    "metric": "auc",
    "first_metric_only": True,
    "max_depth": 6,
    "num_leaves": 56,
    "max_bin": 511,
    "min_data_in_leaf": 1500,
    "learning_rate": 1e-2,
    "feature_fraction": 0.15,
    "subsample": 0.5,
    "subsample_freq": 1,
    "lambda_l2": 0.95,
    "lambda_l1": 0.5,
    "min_gain_to_split": 1e-5,
}

NUM_BOOST_ROUND = 2000

predictions = []
scores = []

for k in range(N_SPLITS):
    print(f"=== Fold {k + 1} ===")
    X_fold, X_val, y_fold, y_val = tools.load(f"3a-engineering/folds/{k}")
    X_fold = X_fold[feature_cols]
    X_val = X_val[feature_cols]

    y_fold_bin = (y_fold["target"].values > 0).astype(int)
    y_val_bin = (y_val["target"].values > 0).astype(int)

    w_fold = np.abs(y_fold["target"].values)
    w_val = np.abs(y_val["target"].values)

    train_data = lgbm.Dataset(X_fold, label=y_fold_bin, weight=w_fold, categorical_feature=cat_cols)
    val_data = lgbm.Dataset(X_val, label=y_val_bin, weight=w_val, categorical_feature=cat_cols)

    evals = {}
    model = lgbm.train(
        lgbm_params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgbm.early_stopping(300, min_delta=1e-4),
            lgbm.log_evaluation(200),
            lgbm.record_evaluation(evals),
        ],
    )

    train_metric = evals["training"]["auc"]
    valid_metric = evals["valid_1"]["auc"]
    plt.figure(figsize=(6, 3))
    plt.plot(train_metric, label="train")
    plt.plot(valid_metric, label="validation")
    plt.axvline(model.best_iteration, linestyle="--", label="best_iter")
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title(f"Fold {k + 1}")
    plt.legend()
    plt.show()

    # SHAP importance
    shap_contrib = model.predict(
        X_val.values, num_threads=lgbm_params["num_threads"], pred_contrib=True
    )
    shap_importance = np.abs(shap_contrib[:, :-1]).mean(axis=0)

    train_pred = model.predict(X_fold.values, num_threads=lgbm_params["num_threads"])
    val_pred = model.predict(X_val.values, num_threads=lgbm_params["num_threads"])

    predictions.append(
        {
            "model": model,
            "best_iter": model.best_iteration,
            ("train", "true"): y_fold["target"].values,
            ("train", "pred"): train_pred,
            ("val", "true"): y_val["target"].values,
            ("val", "pred"): val_pred,
            ("val", "RET_1"): X_val["RET_1"].values,
            "shap_importance": shap_importance,
        }
    )

    scores.append(
        {"fold": k + 1}
        | tools.compute_metrics(
            train=(y_fold["target"].values, train_pred),
            val=(y_val["target"].values, val_pred),
            threshold=0.5,
        )
    )


# %%
print(tools.metrics_table(scores, baseline={("val", "acc"): 0.5189}))

# %%
# Threshold optimization on binary probabilities
y_true_all = np.concatenate([r[("val", "true")] for r in predictions])
y_pred_all = np.concatenate([r[("val", "pred")] for r in predictions])

best_t, best_acc = tools.plot_threshold_optimization(y_true_all, y_pred_all)

# %%
# Accuracy by prediction bin
bin_data = tools.accuracy_by_bin(y_true_all, y_pred_all, n_bins=30, threshold=0.5)
tools.plot_accuracy_by_bin(bin_data, title="Accuracy per predicted probability bin")
plt.show()
tools.print_accuracy_table(bin_data, y_pred_all)

# %%
importances = pd.DataFrame(
    [pred["shap_importance"] for pred in predictions],
    columns=feature_cols,
)
tools.format_importances(importances)

# %%
if PREDICT:
    X_train_full = X_train[feature_cols].copy()
    X_test_full = X_test[feature_cols].copy()

    avg_best_iter = int(np.mean([p["best_iter"] for p in predictions]))
    print(f"Using avg_best_iter = {avg_best_iter} (from {len(predictions)} folds)")

    train_data = lgbm.Dataset(
        X_train_full,
        label=y_train_bin,
        weight=np.abs(y_train["target"].values),
        categorical_feature=cat_cols,
    )
    final_model = lgbm.train(
        lgbm_params,
        train_data,
        num_boost_round=avg_best_iter,
        valid_sets=[train_data],
        callbacks=[lgbm.log_evaluation(200)],
    )

    preds_proba = final_model.predict(
        X_test_full.values,
        num_threads=lgbm_params["num_threads"],
    )

    preds_sign = (preds_proba > 0.5).astype(int)

    # Override low-confidence predictions with RET_1 sign
    if DO_OVERRIDE:
        override_mask = (preds_proba >= LOW_CONF_LOW) & (preds_proba <= LOW_CONF_HIGH)
        ret1_sign = (X_test["RET_1"].values > 0).astype(int)
        n_overridden = override_mask.sum()
        preds_sign[override_mask] = ret1_sign[override_mask]
        print(
            f"Overrode {n_overridden} predictions ({n_overridden / len(preds_proba):.1%})"
            f" with RET_1 sign"
        )

    preds_df = pd.DataFrame({"ROW_ID": X_test.index, "target": preds_sign})

    tools.submit("lightgbm_binary_weighted", preds_df)

# %% [markdown]
# ## Results
#
# ### Comparison with MSE model (4b)
# | Model | Objective | Weighted | CV Val Acc |
# |---|---|---|---|
# | RET_1 sign heuristic | — | — | 51.89% |
# | Ridge (4a) | L2 | — | 52.25% |
# | LightGBM MSE (4b) | L2 | — | 52.55% |
# | **LightGBM Binary (4d)** | **LogLoss** | **\|target\|** | **?** |
#
# ### Key differences from the MSE model
# - **Objective**: `binary` (log-loss) instead of `mse` — directly optimizes the sign prediction task
# - **Sample weighting**: each sample is weighted by `\|target\|`, so high-amplitude returns
#   (where the signal-to-noise ratio is highest) dominate the gradient. Near-zero returns
#   (pure noise) contribute almost nothing.
# - **Low-confidence override**: CV predictions with accuracy below 51% are identified,
#   and test predictions in that probability range are replaced with RET_1 sign (~51.9% heuristic).
#   This prevents the model from making random-guess predictions in the no-signal region.
# - **Metric**: `auc` instead of `mse` — measures ranking quality
# - **boost_from_average**: `True` — initializes with the log-odds prior (~0.028 for 50.7% positive rate)
#
# ### What to try next
# - **Ensemble MSE + Binary**: Blend predictions — low correlation expected between models.
# - **Day-neutral target**: Remove the day mean from returns *before* binarizing.
# - **Soft weight variants**: Try `sqrt(\|target\|)` or clipped weights to reduce extreme sample dominance.
#
