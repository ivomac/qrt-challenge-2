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
# # LightGBM Huber — Optuna Hyperparameter Tuning
#
# **Purpose**: Use Optuna to find optimal LightGBM hyperparameters for the Huber loss
# model, using features from `3c-filtered` and 12-fold cross-validation.
#
# **Approach**:
# - 200 Optuna trials with TPE sampler, each trial runs full 12-fold CV
# - Tuned: learning_rate, max_depth, num_leaves, min_data_in_leaf, feature_fraction,
#   subsample, lambda_l1/l2, min_gain_to_split, alpha
# - Early stopping within each fold (150 rounds no improvement)
# - Uses a manually curated subset of ~140 features rather than all ~4000
# - Best trial retrained on full training data for test submission
#
# **Results**:
# - Best balanced accuracy on CV val: tracked per trial in SQLite database
# - Trial history saved to `6c-fitting-lightgbm_optuna.db` for later analysis

# %%
import tools

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score

import optuna

import plotly.io as pio

pio.renderers.default = "notebook"

sns.set_theme()

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
N_SPLITS = 12
PREDICT = True

OPTUNA_N_TRIALS = 200
OPTUNA_SEED = 42

EARLY_STOP_MIN_DELTA = 1e-7
EARLY_STOP_ROUNDS = 150
NUM_BOOST_ROUND = 5000

# %% [markdown]
# ## Load data & identify features
#
# Uses the correlation-filtered 3b data with all features except ID columns
# (same setup as 6b, no manual exclusions).

# %%
train, test = tools.load("3c-filtered/full")
all_cols = list(train.columns)

exclude = {"TS", "ALLOC"}

feature_cols = [c for c in train.columns if c not in exclude]
cat_cols = [c for c in feature_cols if isinstance(train[c].dtype, pd.CategoricalDtype)]

# %%
selected_cols = {
    "ratio_row_sharpe_RET_4_alloc_mean_row_sharpe_RET_4",
    "ts_max_row_std_RET_20",
    "ratio_row_std_RET_4_ts_std_TURNOVER",
    "ts_mean_row_sharpe_SVOLUME_20",
    "ts_mean_row_sharpe_VOLUME_4",
    "ts_min_row_std_VOLUME_20",
    "ts_min_row_std_RET_4",
    "ts_max_VOLUME_16",
    "ts_min_RET_17",
    "ts_min_RET_15",
    "ts_max_row_mean_RET_14",
    "ts_min_RET_18",
    "ts_mean_VOLUME_16",
    "ts_max_SVOLUME_17",
    "ts_max_SVOLUME_1",
    "ts_max_SVOLUME_12",
    "ts_mean_row_std_VOLUME_14",
    "ts_min_RET_11",
    "ts_std_row_mean_VOLUME_20",
    "ts_max_SVOLUME_4",
    "alloc_std_TURNOVER",
    "ts_std_VOLUME_8",
    "ts_std_RET_9",
    "ts_mean_TURNOVER",
    "ts_min_row_sharpe_SVOLUME_4",
    "ts_min_RET_2",
    "ts_min_VOLUME_2",
    "ts_std_row_sharpe_SVOLUME_20",
    "ts_std_row_mean_RET_14",
    "ts_mean_VOLUME_19",
    "ts_std_row_sharpe_RET_14",
    "ts_min_row_sharpe_SVOLUME_7",
    "ts_std_RET_2",
    "ts_std_SVOLUME_15",
    "ts_std_RET_13",
    "ts_max_VOLUME_3",
    "row_weighted_sum_RET_pos_7",
    "diff_ts_pct_rank_RET_1_row_mean_ts_pct_rank_RET_4",
    "ts_max_RET_8",
    "ts_std_RET_4",
    "ts_mean_SVOLUME_11",
    "ts_max_VOLUME_8",
    "ts_max_RET_1",
    "ts_std_RET_1",
    "ratio_row_sharpe_RET_14_ts_mean_row_sharpe_RET_14",
    "ts_std_RET_3",
    "ts_max_RET_13",
    "ratio_row_mean_RET_neg_14_TURNOVER",
    "ratio_row_mean_ts_zscore_RET_4_TURNOVER",
    "ts_min_row_std_RET_7",
    "ts_max_VOLUME_17",
    "ratio_RET_VOLUME_7",
    "ts_min_row_sharpe_RET_20",
    "ts_std_row_sharpe_RET_4",
    "ts_std_row_sharpe_SVOLUME_7",
    "ratio_row_mean_ratio_RET_VOLUME_4_TURNOVER",
    "alloc_std_row_sharpe_VOLUME_7",
    "SVOLUME_neg_11",
    "ratio_RET_1_row_mean_RET_14",
    "ratio_row_mean_SVOLUME_14_TURNOVER",
    "ts_std_row_sharpe_VOLUME_7",
    "ts_std_RET_11",
    "ts_mean_row_std_VOLUME_7",
    "RET_7",
    "ts_mean_row_std_VOLUME_4",
    "ts_std_row_sharpe_SVOLUME_4",
    "ts_std_row_std_RET_14",
    "ts_std_RET_7",
    "ratio_row_std_RET_4_alloc_mean_row_std_RET_4",
    "ts_max_row_sharpe_SVOLUME_4",
    "ts_std_SVOLUME_20",
    "ratio_row_mean_ts_pct_rank_SVOLUME_14_TURNOVER",
    "ts_min_RET_8",
    "ts_max_SVOLUME_16",
    "ts_mean_row_sharpe_RET_7",
    "ratio_RET_pos_1_ts_mean_TURNOVER",
    "ts_min_row_sharpe_RET_14",
    "ts_max_row_std_RET_14",
    "ts_mean_row_sharpe_SVOLUME_7",
    "ts_std_VOLUME_16",
    "ts_mean_SVOLUME_20",
    "ratio_RET_neg_1_ts_mean_TURNOVER",
    "ts_std_RET_8",
    "ts_mean_row_sharpe_SVOLUME_4",
    "ts_mean_VOLUME_2",
    "ratio_RET_1_row_mean_RET_20",
    "ts_std_RET_10",
    "ts_mean_row_sharpe_VOLUME_7",
    "ts_demean_TURNOVER",
    "RET_pos_7",
    "row_mean_sqrt_RET_pos_20",
    "ts_min_row_mean_VOLUME_4",
    "ts_min_row_mean_VOLUME_14",
    "ratio_row_std_SVOLUME_pos_14_TURNOVER",
    "ts_std_row_mean_RET_20",
    "row_weighted_sum_RET_pos_14",
    "ts_max_RET_20",
    "ts_std_row_mean_SVOLUME_14",
    "ts_mean_row_std_RET_7",
    "ts_max_TURNOVER",
    "ts_mean_row_std_RET_20",
    "row_mean_sqrt_RET_pos_14",
    "alloc_mean_TURNOVER",
    "sq_RET_1",
    "ts_min_SVOLUME_20",
    "ratio_row_mean_RET_pos_20_ts_std_TURNOVER",
    "ts_min_SVOLUME_10",
    "ts_min_row_std_SVOLUME_14",
    "ts_std_RET_15",
    "ts_min_row_mean_VOLUME_7",
    "ratio_row_sharpe_RET_20_ts_mean_row_sharpe_RET_20",
    "ts_min_VOLUME_20",
    "ts_max_VOLUME_20",
    "ts_max_row_sharpe_SVOLUME_7",
    "diff_RET_neg_1_row_mean_RET_neg_20",
    "diff_RET_1_row_mean_RET_7",
    "ts_min_row_std_RET_20",
    "ts_mean_row_mean_RET_7",
    "ts_max_row_sharpe_RET_20",
    "ts_max_row_sharpe_SVOLUME_14",
    "ratio_RET_neg_1_row_mean_RET_neg_4",
    "ts_std_row_std_RET_4",
    "ts_mean_RET_1",
    "ratio_row_mean_RET_pos_14_ts_std_TURNOVER",
    "diff_RET_neg_1_row_mean_RET_neg_4",
    "ratio_RET_1_ts_mean_TURNOVER",
    "ratio_RET_neg_1_TURNOVER",
    "ts_max_row_sharpe_VOLUME_20",
    "ts_min_row_std_RET_14",
    "ratio_row_mean_prod_RET_VOLUME_4_TURNOVER",
    "ratio_RET_pos_1_row_mean_RET_pos_4",
    "ts_mean_RET_17",
    "ratio_RET_1_TURNOVER",
    "ts_mean_row_std_RET_4",
    "alloc_sharpe_row_mean_RET_4",
    "alloc_sharpe_row_sharpe_RET_14",
    "diff_RET_pos_1_row_mean_RET_pos_4",
    "alloc_sharpe_row_sharpe_RET_4",
    "diff_RET_1_row_mean_RET_4",
}

# %%
feature_cols = [feat for feat in feature_cols if feat in selected_cols]
cat_cols = [cat for cat in cat_cols if cat in selected_cols]

# %%
sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=6)
study = optuna.create_study(
    direction="maximize",
    study_name="huber",
    storage="sqlite:///6c-fitting-lightgbm_optuna.db",
    sampler=sampler,
    pruner=pruner,
    load_if_exists=True,
)

# %%
BASE_PARAMS = {
    "num_threads": 12,
    "verbose": -1,
    "seed": 42,
    "max_cat_to_onehot": 5,
    "objective": "huber",
    "metric": "huber",
    "max_bin": 511,
    "max_depth": 8,
    "subsample_freq": 1,
    "min_gain_to_split": 1e-9,
}


def objective(trial: optuna.Trial) -> float:
    params = BASE_PARAMS | {
        "num_leaves": trial.suggest_int("num_leaves", 56, 96, step=8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 800, 3200, step=1200),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 5e-2, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 6e-3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.05, 0.95, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 50, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.001, 10),
    }

    val_accs = []
    train_accs = []
    for k in range(N_SPLITS):
        X_fold, X_val = tools.load(f"3c-filtered/folds/{k}", verbose=False)
        X_fold = X_fold[feature_cols]
        X_val = X_val[feature_cols]

        train_data = lgbm.Dataset(
            X_fold, label=X_fold["RET_0"].values, categorical_feature=cat_cols
        )
        val_data = lgbm.Dataset(X_val, label=X_val["RET_0"].values, categorical_feature=cat_cols)

        model = lgbm.train(
            params,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_data],
            callbacks=[
                lgbm.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgbm.log_evaluation(-1),
            ],
        )

        train_pred = model.predict(X_fold.values, num_threads=params["num_threads"])
        val_pred = model.predict(X_val.values, num_threads=params["num_threads"])

        train_acc = balanced_accuracy_score(
            (X_fold["RET_0"].values > 0).astype(int),
            (train_pred > 0).astype(int),
        )
        val_acc = balanced_accuracy_score(
            (X_val["RET_0"].values > 0).astype(int),
            (val_pred > 0).astype(int),
        )
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        trial.report(float(np.mean(val_accs)), step=k)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("train_acc_mean", float(np.mean(train_accs)))
    trial.set_user_attr("train_acc_std", float(np.std(train_accs)))
    trial.set_user_attr("val_acc_mean", float(np.mean(val_accs)))
    trial.set_user_attr("val_acc_std", float(np.std(val_accs)))
    trial.set_user_attr("gap", float(np.mean(train_accs) - np.mean(val_accs)))

    return float(np.mean(val_accs))


study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)

# %% [markdown]
# ## Best trial results

# %%
clean_study = optuna.create_study(direction=study.direction)
clean_study.add_trials(
    [
        t
        for t in study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        if t.value is not None and t.value >= 0.525
    ]
)

fig = optuna.visualization.plot_optimization_history(clean_study)
fig.show()

fig2 = optuna.visualization.plot_param_importances(clean_study)
fig2.show()

fig3 = optuna.visualization.plot_slice(clean_study)
fig3.show()

# %%
print(f"Best trial: {study.best_trial.number}")
print(f"Best balanced accuracy: {study.best_value:.4f}")

df = study.trials_dataframe().sort_values(by="user_attrs_val_acc_mean", ascending=False)

prefixes = ["params_", "user_attrs_"]


def renamer(col):
    for prefix in prefixes:
        if isinstance(col, str) and col.startswith(prefix):
            col = col[len(prefix) :]
    return col


cols = [col for col in df for pref in prefixes if col.startswith(pref)]
print(df[cols].rename(columns=renamer).to_string(float_format=lambda x: f"{x:.2e}"))

# %% [markdown]
# ## Retrain with best params

# %%
best_params = BASE_PARAMS | study.best_params

predictions = []
scores = []
gains = []

for k in range(N_SPLITS):
    print(f"=== Fold {k + 1} ===")
    X_fold, X_val = tools.load(f"3c-filtered/folds/{k}")
    X_fold = X_fold[feature_cols]
    X_val = X_val[feature_cols]

    train_data = lgbm.Dataset(X_fold, label=X_fold["RET_0"].values, categorical_feature=cat_cols)
    val_data = lgbm.Dataset(X_val, label=X_val["RET_0"].values, categorical_feature=cat_cols)

    evals = {}
    model = lgbm.train(
        best_params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgbm.early_stopping(EARLY_STOP_ROUNDS, min_delta=EARLY_STOP_MIN_DELTA),
            lgbm.log_evaluation(200),
            lgbm.record_evaluation(evals),
        ],
    )

    train_metric = evals["training"]["huber"]
    valid_metric = evals["valid_1"]["huber"]
    plt.figure(figsize=(6, 3))
    plt.plot(train_metric, label="train")
    plt.plot(valid_metric, label="validation")
    plt.axvline(model.best_iteration, linestyle="--", label="best_iter")
    plt.xlabel("Iteration")
    plt.ylabel("Huber Loss")
    plt.title(f"Fold {k + 1}")
    plt.legend()
    plt.show(block=False)

    train_pred = model.predict(X_fold.values, num_threads=best_params["num_threads"])
    val_pred = model.predict(X_val.values, num_threads=best_params["num_threads"])

    gain = dict(
        zip(
            model.feature_name(),
            model.feature_importance(importance_type="gain"),
        )
    )
    gains.append(gain)

    predictions.append(
        {
            "model": model,
            "best_iter": model.best_iteration,
            ("val", "true"): X_val["RET_0"].values,
            ("val", "pred"): val_pred,
            ("train", "true"): X_fold["RET_0"].values,
            ("train", "pred"): train_pred,
        }
    )

    row = {"fold": k + 1}
    metrics = tools.compute_metrics(
        train=(X_fold["RET_0"].values, train_pred),
        val=(X_val["RET_0"].values, val_pred),
    )
    row.update(metrics)
    scores.append(row)

# %%
print(tools.metrics_table(scores))

# %%
# Gain importance across folds
importances = pd.DataFrame([pd.Series(g) for g in gains]).fillna(0.0)
tools.format_importances(importances, sort_col="ratio")

# %%
# Threshold optimization
all_y_true = np.concatenate([p[("val", "true")] for p in predictions])
all_y_pred = np.concatenate([p[("val", "pred")] for p in predictions])
tools.plot_threshold_optimization(all_y_true, all_y_pred)

# %%
# Accuracy by prediction bin
bin_data = tools.accuracy_by_bin(all_y_true, all_y_pred, n_bins=30)
tools.plot_accuracy_by_bin(bin_data, title="Accuracy per predicted return bin")
plt.show(block=False)

tools.print_accuracy_table(bin_data, all_y_pred)

# %% [markdown]
# ## Final model & submission

# %%
if PREDICT:
    X_train_full = train[feature_cols].copy()
    X_test_full = test[feature_cols].copy()

    avg_best_iter = int(np.mean([p["best_iter"] for p in predictions]))
    print(f"Using avg_best_iter = {avg_best_iter} (from {len(predictions)} folds)")

    train_data = lgbm.Dataset(
        X_train_full,
        label=train["RET_0"].values,
        categorical_feature=cat_cols,
    )
    final_model = lgbm.train(
        best_params,
        train_data,
        num_boost_round=avg_best_iter,
        valid_sets=[train_data],
        callbacks=[lgbm.log_evaluation(200)],
    )

    preds = final_model.predict(
        X_test_full.values,
        num_threads=best_params["num_threads"],
    )

    sub = pd.DataFrame(
        {
            "ROW_ID": test.index,
            "target": (preds > 0).astype(int),
        }
    )

    tools.submit("lightgbm_huber_optuna", sub)
