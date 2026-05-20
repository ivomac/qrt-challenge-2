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
# # LightGBM Huber — Iterative Feature Selection
#
# **Purpose**: Train a LightGBM model with Huber loss on correlation-filtered features,
# with an iterative feature selection loop that drops low-importance features each round.
#
# **Approach**:
# - 12-fold TS-based cross-validation on features from `3c-filtered`
# - Greedy feature pruning: after each round, drop features with low gain importance
#   (mean/std ratio < 0.8) or very small contribution (< 0.5/N % of total gain)
# - Stops when val accuracy drops or standard deviation increases beyond tolerance
# - Final submission uses median of fold-level test predictions
#
# **Results**:
# - Top features are dominated by RET_1-derived columns and TS-level statistics

# %%
import tools

import ctypes
import gc
import json
import time
from pathlib import Path

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

_libc = ctypes.cdll.LoadLibrary("libc.so.6")

# %%
N_SPLITS = 12

EARLY_STOP_MIN_DELTA = 1e-7
EARLY_STOP_ROUNDS = 150
NUM_BOOST_ROUND = 5000
LOG_EVERY = 200

MAX_ROUNDS = 5
IMPORTANCE_SCALE = 0.5
RATIO_THRESHOLD = 0.8
VAL_ACC_DROP_TOL = 0.0001
VAL_STD_INCREASE_TOL = 0.0006

lgbm_params = {
    "num_threads": 12,
    "verbose": -1,
    "seed": 42,
    "boost_from_average": False,
    "max_cat_to_onehot": 5,
    "objective": "huber",
    "alpha": 1e-1,
    "metric": "huber",
    "max_depth": 6,
    "num_leaves": 56,
    "max_bin": 511,
    "min_data_in_leaf": 800,
    "learning_rate": 1.4e-2,
    "feature_fraction": 0.2,
    "subsample": 0.7,
    "subsample_freq": 1,
    "lambda_l2": 2,
    "lambda_l1": 0.1,
    "min_gain_to_split": 1e-6,
}

# %% [markdown]
# ## Pipeline
#
# Iterative feature selection via gain importance:
# 1. Train all folds with current features, measure per-fold gain importance
# 2. Drop features where pct < A OR ratio (mean/std) < B
# 3. Stop if val acc drops > C or val std increases > D, or no features dropped

# %%
test = pd.read_parquet("../data/3c-filtered/full/test.parquet")

# %%
exclude = {"TS", "ALLOC", "ROW_ID", "RET_0"}

feature_cols = [c for i, c in enumerate(test.columns) if c not in exclude]
cat_cols = [c for c in feature_cols if isinstance(test[c].dtype, pd.CategoricalDtype)]

# %% [markdown]
# ## Iterative feature selection

# %%
current_features = list(feature_cols)
current_cat_cols = list(cat_cols)
round_results: list[dict] = []

for round_idx in range(MAX_ROUNDS):
    t_round_start = time.time()
    print(f"\n{'=' * 60}")
    print(
        f"ROUND {round_idx + 1}/{MAX_ROUNDS}  —  {len(current_features)} features, {len(current_cat_cols)} cat"
    )
    print(f"{'=' * 60}")

    scores = []
    gains = []
    best_iters = []
    fold_times = []
    predictions = []
    X_test_sub = test[current_features].values

    for k in range(N_SPLITS):
        print(f"-- Fold {k + 1} --")
        X_fold, X_val = tools.load(f"3c-filtered/folds/{k}")
        X_fold_sub = X_fold[current_features]
        X_val_sub = X_val[current_features]

        t0 = time.time()

        train_data = lgbm.Dataset(
            X_fold_sub.values,
            label=X_fold["RET_0"].values,
            feature_name=current_features,
            categorical_feature=current_cat_cols,
        )
        val_data = lgbm.Dataset(
            X_val_sub.values,
            label=X_val["RET_0"].values,
            feature_name=current_features,
            categorical_feature=current_cat_cols,
        )

        model = lgbm.train(
            lgbm_params,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[train_data, val_data],
            keep_training_booster=True,
            callbacks=[
                lgbm.early_stopping(EARLY_STOP_ROUNDS, min_delta=EARLY_STOP_MIN_DELTA),
                lgbm.log_evaluation(LOG_EVERY),
            ],
        )

        elapsed = time.time() - t0
        fold_times.append(elapsed)

        train_pred = model.predict(X_fold_sub.values, num_threads=lgbm_params["num_threads"])
        val_pred = model.predict(X_val_sub.values, num_threads=lgbm_params["num_threads"])
        test_pred = model.predict(X_test_sub, num_threads=lgbm_params["num_threads"])

        gain = dict(
            zip(
                model.feature_name(),
                model.feature_importance(importance_type="gain"),
            )
        )
        gains.append(gain)
        best_iters.append(model.best_iteration)

        predictions.append(
            {
                "best_iter": model.best_iteration,
                ("train", "pred"): train_pred,
                ("val", "pred"): val_pred,
                "test_preds": test_pred,
            }
        )

        scores.append(
            {"fold": k + 1}
            | tools.compute_metrics(
                train=(X_fold["RET_0"].values, train_pred),
                val=(X_val["RET_0"].values, val_pred),
            )
        )

        del model, train_data, val_data, X_fold, X_val, X_fold_sub, X_val_sub, train_pred, val_pred
        gc.collect()
        _libc.malloc_trim(0)

    # --- Aggregate ---
    val_accs = [s[("val", "acc")] for s in scores]
    val_acc_mean = np.mean(val_accs)
    val_acc_std = np.std(val_accs)
    train_acc_mean = np.mean([s[("train", "acc")] for s in scores])
    avg_fold_time = np.mean(fold_times)

    gains_df = pd.DataFrame([pd.Series(g) for g in gains]).fillna(0.0)
    del gains
    mean_gain = gains_df.mean()
    std_gain = gains_df.std()
    total_gain = mean_gain.sum()
    pct = mean_gain / total_gain * 100
    ratio = mean_gain / std_gain

    round_results.append(
        {
            "round": round_idx + 1,
            "n_features": len(current_features),
            "n_cat": len(current_cat_cols),
            "val_acc_mean": val_acc_mean,
            "val_acc_std": val_acc_std,
            "train_acc_mean": train_acc_mean,
            "avg_fold_time": avg_fold_time,
            "avg_best_iter": float(np.mean(best_iters)),
            "features": current_features.copy(),
            "cat_cols": current_cat_cols.copy(),
            "scores": scores,
            "predictions": predictions,
            "gains_df": gains_df,
            "pct": pct,
            "ratio": ratio,
        }
    )

    print(f"\n--- Round {round_idx + 1} Summary ---")
    print(f"Val acc:    {val_acc_mean:.4%} ± {val_acc_std:.4%}")
    print(f"Train acc:  {train_acc_mean:.4%}")
    print(f"Avg fold time: {avg_fold_time:.1f}s")
    print(f"Avg best_iter: {round_results[-1]['avg_best_iter']:.0f}")
    print("Top 10 features by gain pct:")
    for f in pct.sort_values(ascending=False).head(10).index:
        print(f"  {pct[f]:6.2%}  ratio={ratio[f]:5.2f}  {f}")

    # --- Break conditions ---
    if round_idx > 0:
        prev = round_results[-2]
        acc_drop = prev["val_acc_mean"] - val_acc_mean
        std_inc = val_acc_std - prev["val_acc_std"]
        if acc_drop > VAL_ACC_DROP_TOL:
            print(f"\nSTOP: Val acc dropped {acc_drop:.4%} > {VAL_ACC_DROP_TOL:.4%}")
            break
        if std_inc > VAL_STD_INCREASE_TOL:
            print(f"\nSTOP: Val std increased {std_inc:.4%} > {VAL_STD_INCREASE_TOL:.4%}")
            break

    # --- Feature selection ---
    pct_threshold = 100 * IMPORTANCE_SCALE / len(current_features)
    keep = [
        f
        for f in current_features
        if pct.get(f, 0.0) > pct_threshold and ratio.get(f, 0.0) > RATIO_THRESHOLD
    ]

    n_dropped = len(current_features) - len(keep)
    print(
        f"\nDropping {n_dropped} features (pct ≤ {pct_threshold:.3f}% or ratio ≤ {RATIO_THRESHOLD})"
    )

    if n_dropped == 0:
        print("Converged — no features to drop.")
        break

    current_features = keep
    current_cat_cols = [c for c in cat_cols if c in current_features]

    del scores, fold_times, best_iters, X_test_sub
    gc.collect()
    _libc.malloc_trim(0)

print(f"\n{'=' * 60}")
print(f"Feature selection complete after {len(round_results)} round(s)")
print(f"{'=' * 60}")

# %% [markdown]
# ## Results

# %%
round_summary = pd.DataFrame(
    [
        {
            "Round": r["round"],
            "Features": r["n_features"],
            "Val Acc": r["val_acc_mean"],
            "Val Std": r["val_acc_std"],
            "Train Acc": r["train_acc_mean"],
            "Avg Time (s)": r["avg_fold_time"],
            "Avg Best Iter": r["avg_best_iter"],
        }
        for r in round_results
    ]
)
round_summary.index = round_summary["Round"]
round_summary = round_summary.drop(columns=["Round"])
print(
    round_summary.map(
        lambda x: (
            f"{x:4.2%}"
            if isinstance(x, float) and abs(x) < 1
            else (
                f"{x:.1f}s" if "Time" in str(x) else f"{x:.0f}" if isinstance(x, float) else str(x)
            )
        )
    ).to_string()
)

# %%
sorted_rounds = sorted(round_results, key=lambda r: r["val_acc_mean"], reverse=True)
best = sorted_rounds[0]
print(f"Best round: {best['round']}  (val acc = {best['val_acc_mean']:.4%})")

predictions = best["predictions"]
scores = best["scores"]
feature_cols_best = best["features"]

# %% [markdown]
# ### Metrics
print(tools.metrics_table(scores))

# %% jupyter={"outputs_hidden": false}
# Threshold optimization
y_true_all = np.concatenate([r[("val", "true")] for r in predictions])
y_pred_all = np.concatenate([r[("val", "pred")] for r in predictions])

best_t, best_acc = tools.plot_threshold_optimization(y_true_all, y_pred_all)

# %% jupyter={"outputs_hidden": false}
# Accuracy by prediction bin
bin_data = tools.accuracy_by_bin(y_true_all, y_pred_all, n_bins=30, threshold=0)
tools.plot_accuracy_by_bin(bin_data, title="Accuracy per predicted return bin")
plt.show(block=False)

tools.print_accuracy_table(bin_data, y_pred_all)

# %% jupyter={"outputs_hidden": false}
# Gain importance across folds
importances = best["gains_df"]
tools.format_importances(importances, sort_col="ratio")

# %%
# Save selected feature list for use by prediction script

out = Path("../data/6b-features")
out.mkdir(parents=True, exist_ok=True)
(out / "feature_cols.json").write_text(json.dumps(feature_cols_best))
print(f"Saved {len(feature_cols_best)} features to {out}")

# %% [markdown]
# ## Submission

# %%
last_round = round_results[-1]
best_round = sorted(round_results, key=lambda r: r["val_acc_mean"], reverse=True)[0]

for label, r in [("last", last_round), ("best", best_round)]:
    pred_matrix = np.stack(r["test_preds"], axis=0)
    median_pred = np.median(pred_matrix, axis=0)
    submission = pd.DataFrame(
        {
            "ROW_ID": test["ROW_ID"],
            "prediction": (median_pred > 0).astype(np.int64),
        }
    )
    name = f"6b_lgbm_huber_{label}_round{r['round']}"
    tools.submit(name, submission)
