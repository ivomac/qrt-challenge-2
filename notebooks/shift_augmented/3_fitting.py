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
# # LightGBM Huber — Shift-Augmented Fitting
#
# **Purpose**: Train LightGBM with Huber loss on shift-augmented features. A single-pass
# 12-fold CV with median ensemble on test predictions.
#
# **Approach**:
# - Reads shift-augmented train and unshifted validation from `4d-features/folds/{k}`
# - Target `RET_0` is embedded in X (SHIFT=0 rows only)
# - No iterative feature pruning — all 24 engineered features used directly
# - Each fold model predicts on full test set; final prediction is the median sign
#   across all folds
#
# **Results**: This alternate approach did not beat the main pipeline. The shift
# augmentation introduces synthetic rows that may not reflect real temporal dynamics.

# %%
import tools

import time

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

# %%
N_SPLITS = 12

NON_FEATURES = {"ROW_ID", "TS", "ALLOC", "GROUP", "SHIFT", "RET_0"}

lgbm_params = {
    "num_threads": 10,
    "verbose": -1,
    "seed": 42,
    "boost_from_average": False,
    "max_cat_to_onehot": 5,
    "objective": "huber",
    "metric": "huber",
    "alpha": 1e-3,
    "max_depth": 6,
    "num_leaves": 56,
    "max_bin": 511,
    "min_data_in_leaf": 2000,
    "learning_rate": 5e-2,
    "feature_fraction": 0.5,
    "subsample": 0.7,
    "subsample_freq": 1,
    "lambda_l2": 3,
    "lambda_l1": 0.1,
    "min_gain_to_split": 1e-8,
}
NUM_BOOST_ROUND = 10000
EARLY_STOP_ROUNDS = 150
LOG_EVERY = 200

# %%
# Load full dataset — needed for test ROW_IDs and final model
X_train_full, X_test_full = tools.load("4d-features/full")
feature_cols = [c for c in X_train_full.columns if c not in NON_FEATURES]
print(f"Features: {len(feature_cols)}")
print(f"Full train rows: {len(X_train_full):,}  |  Test rows: {len(X_test_full):,}")

# %% [markdown]
# ## CV loop

# %%
predictions = []
scores = []
gains = []
best_iters = []
test_preds_per_fold = []
evals_per_fold = []

for k in range(N_SPLITS):
    X_fold, X_val = tools.load(f"4d-features/folds/{k}")

    y_fold = X_fold["RET_0"].values
    y_val = X_val["RET_0"].values

    X_fold_f = X_fold[feature_cols]
    X_val_f = X_val[feature_cols]
    X_test_f = X_test_full[feature_cols]

    t0 = time.time()

    train_data = lgbm.Dataset(X_fold_f, label=y_fold)
    val_data = lgbm.Dataset(X_val_f, label=y_val, reference=train_data)

    evals = {}
    model = lgbm.train(
        lgbm_params,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgbm.early_stopping(EARLY_STOP_ROUNDS),
            lgbm.log_evaluation(LOG_EVERY),
            lgbm.record_evaluation(evals),
        ],
    )
    evals_per_fold.append(evals)

    train_loss = evals["training"]["huber"]
    val_loss = evals["valid_1"]["huber"]
    iters_k = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(iters_k, train_loss, label="train")
    ax.plot(iters_k, val_loss, label="val")
    ax.axvline(
        model.best_iteration,
        linestyle="--",
        color="gray",
        label=f"best_iter={model.best_iteration}",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Huber loss")
    ax.set_title(f"Fold {k}")
    ax.legend()
    plt.show(block=False)

    elapsed = time.time() - t0
    best_iters.append(model.best_iteration)

    train_pred = model.predict(X_fold_f.values, num_threads=lgbm_params["num_threads"])
    val_pred = model.predict(X_val_f.values, num_threads=lgbm_params["num_threads"])
    test_pred = model.predict(X_test_f.values, num_threads=lgbm_params["num_threads"])

    test_preds_per_fold.append(test_pred)

    gain = dict(zip(model.feature_name(), model.feature_importance(importance_type="gain")))
    gains.append(gain)

    predictions.append(
        {
            "best_iter": model.best_iteration,
            ("train", "true"): y_fold,
            ("train", "pred"): train_pred,
            ("val", "true"): y_val,
            ("val", "pred"): val_pred,
        }
    )

    scores.append(
        {"fold": k + 1}
        | tools.compute_metrics(
            train=(y_fold, train_pred),
            val=(y_val, val_pred),
        )
    )

    val_acc = scores[-1][("val", "acc")]
    print(
        f"fold {k:2d}: train_rows={len(X_fold):>9,}  val_rows={len(X_val):>6,}  "
        f"val_acc={val_acc:.4%}  best_iter={model.best_iteration}  ({elapsed:.0f}s)"
    )

# %% [markdown]
# ## Metrics

# %%
print(tools.metrics_table(scores))

val_accs = [s[("val", "acc")] for s in scores]
train_accs = [s[("train", "acc")] for s in scores]
print(f"\nVal  acc: {np.mean(val_accs):.4%} ± {np.std(val_accs):.4%}")
print(f"Train acc: {np.mean(train_accs):.4%} ± {np.std(train_accs):.4%}")
print(f"Avg best_iter: {np.mean(best_iters):.0f}")

# %%
# Train vs val Huber loss over iterations (mean ± std across folds)
min_iters = min(len(e["training"]["huber"]) for e in evals_per_fold)
train_curves = np.array([e["training"]["huber"][:min_iters] for e in evals_per_fold])
val_curves = np.array([e["valid_1"]["huber"][:min_iters] for e in evals_per_fold])
iters = np.arange(1, min_iters + 1)
avg_best_iter = int(np.mean(best_iters))

fig, ax = plt.subplots(figsize=(9, 4))
for label, curves, color in [
    ("train", train_curves, "steelblue"),
    ("val", val_curves, "darkorange"),
]:
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    ax.plot(iters, mean, label=label, color=color)
    ax.fill_between(iters, mean - std, mean + std, alpha=0.2, color=color)
ax.axvline(avg_best_iter, linestyle="--", color="gray", label=f"avg best_iter={avg_best_iter}")
ax.set_xlabel("Iteration")
ax.set_ylabel("Huber loss")
ax.set_title("Train vs val loss over iterations (mean ± std, 12 folds)")
ax.legend()
plt.show(block=False)

# %%
# Threshold optimization
y_true_all = np.concatenate([p[("val", "true")] for p in predictions])
y_pred_all = np.concatenate([p[("val", "pred")] for p in predictions])

best_t, best_acc = tools.plot_threshold_optimization(y_true_all, y_pred_all)
plt.show(block=False)

# %%
# Accuracy by predicted return bin
bin_data = tools.accuracy_by_bin(y_true_all, y_pred_all, n_bins=30, threshold=0)
tools.plot_accuracy_by_bin(bin_data, title="Accuracy per predicted return bin")
plt.show(block=False)
tools.print_accuracy_table(bin_data, y_pred_all)

# %%
# Feature importance across folds
gains_df = pd.DataFrame([pd.Series(g) for g in gains]).fillna(0.0)
tools.format_importances(gains_df, sort_col="ratio")

# %% [markdown]
# ## Test predictions: median across folds → sign → submit

# %%
test_preds_matrix = np.array(test_preds_per_fold)  # (N_SPLITS, n_test)
median_pred = np.median(test_preds_matrix, axis=0)
final_pred = (median_pred > 0).astype(int)

print(f"Test rows: {len(final_pred)}")
print(f"Predicted positive: {final_pred.sum()} ({final_pred.mean():.2%})")
print(f"Median pred range: [{median_pred.min():.4f}, {median_pred.max():.4f}]")

sub = pd.DataFrame(
    {
        "ROW_ID": X_test_full["ROW_ID"].values,
        "prediction": final_pred,
    }
)

tools.submit("4e-lgbm-4d-features", sub)
