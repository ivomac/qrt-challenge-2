import os
import shutil
import warnings

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "jemalloc")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import balanced_accuracy_score, recall_score

warnings.filterwarnings("ignore", message="FigureCanvas.*is non-interactive")


def save(folder, train, test, verbose=False, chunk_size=20_000):
    folder = Path(f"../data/{folder}")
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("test", test)]:
        if df is None:
            continue
        path = (folder / name).with_suffix(".parquet")
        reset_index = df.index.name is not None
        writer = None
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            if reset_index:
                chunk = chunk.reset_index()
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(path), table.schema, compression="none")
            writer.write_table(table)
        if writer is not None:
            writer.close()
        if verbose:
            mb = df.memory_usage(deep=True).sum() / 1e6
            print(
                f"{name}: {df.shape}, Memory: {mb:.2f} MB, Disk: {path.stat().st_size / 1e6:.2f} MB"
            )


def clean(folder):
    folder = Path(f"../data/{folder}")
    if folder.exists():
        shutil.rmtree(folder)
        print(f"Cleaned: {folder}")
    else:
        print(f"Nothing to clean: {folder}")


def submit(name, df):
    assert list(df.columns) == ["ROW_ID", "prediction"], df.columns
    assert df.shape[0] == 31870
    assert all(r_id == exp for r_id, exp in zip(df["ROW_ID"], range(527073, 558943)))
    assert df["prediction"].isin([0, 1]).all()

    folder = Path("../submissions")
    folder.mkdir(parents=True, exist_ok=True)
    path = (folder / name).with_suffix(".csv")
    df.to_csv(path, index=False)
    print(f"Saved predictions as {path.name}")

    # print prediction statistics:
    print(df["prediction"].value_counts())
    print(f"Fraction positives: {df['prediction'].mean():.2%}")


def load(folder, verbose=False):
    folder = Path(f"../data/{folder}")
    train_path = folder / "train.parquet"
    test_path = folder / "test.parquet"
    train = pd.read_parquet(train_path, engine="pyarrow")
    test = pd.read_parquet(test_path, engine="pyarrow")
    if verbose:
        print(f"train: {train.shape}")
        print(f"test:  {test.shape}")
    return train, test


def load_columns(folder: str, columns: list[str], verbose=False):
    folder_path = Path(f"../data/{folder}")
    train = pd.read_parquet(folder_path / "train.parquet", columns=list(columns))
    test = pd.read_parquet(folder_path / "test.parquet", columns=list(columns))
    if verbose:
        print(f"train: {train.shape}")
        print(f"test:  {test.shape}")
    return train, test


def drop_constant_cols(dfs, uniques=1):
    invalid_cols = [col for col in dfs[0].columns if dfs[0][col].nunique() <= uniques]
    if invalid_cols:
        print(f"Columns to drop: {invalid_cols}")
        for df in dfs:
            df.drop(columns=invalid_cols, inplace=True)


def convert_types(dfs, typemap=None):
    if typemap is None:
        typemap = {}
    typemap = {**{"float64": "float32", "int64": "int32"}, **typemap}

    for df in dfs:
        for curr_type, new_type in typemap.items():
            cols = df.select_dtypes(include=[curr_type]).columns
            if len(cols) > 0:
                df[cols] = df[cols].astype(new_type)


def compute_metrics(train=None, val=None, test=None, threshold=0.0):
    result = {}
    for side, data in [("train", train), ("val", val), ("test", test)]:
        if data is None:
            continue
        y_true, y_pred = data
        y_true_sign = (y_true > 0).astype(int)
        y_pred_sign = (y_pred > threshold).astype(int)
        acc = balanced_accuracy_score(y_true_sign, y_pred_sign)
        r0 = recall_score(y_true_sign, y_pred_sign, pos_label=0)
        r1 = recall_score(y_true_sign, y_pred_sign, pos_label=1)
        rh = 2 * r0 * r1 / (r0 + r1)
        result[(side, "acc")] = acc
        result[(side, "R-0")] = r0
        result[(side, "R-h")] = rh
    return result


def format_importances(
    importances, pct_col="pct", std_col="std_pct", rat_col="ratio", sort_col="pct"
):
    total = importances.mean().sum()
    summary = (
        pd.DataFrame(
            {
                pct_col: importances.mean() / total * 100,
                std_col: importances.std() / total * 100,
                rat_col: importances.mean() / importances.std(),
            }
        )
        .sort_index()
        .sort_values(sort_col)
    )

    def _fmt(x):
        if isinstance(x, str):
            return x
        return f"{x:.2f}%"

    print(summary.to_string(formatters={c: _fmt for c in summary.columns}))


def accuracy_by_bin(y_true, y_pred, n_bins=30, threshold=0.0):
    correct = ((y_true > 0) == (y_pred > threshold)).astype(int)

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_pred, percentiles)
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    bin_indices = np.digitize(y_pred, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, actual_n_bins - 1)

    counts = np.zeros(actual_n_bins, dtype=int)
    correct_counts = np.zeros(actual_n_bins, dtype=int)
    for i in range(actual_n_bins):
        mask = bin_indices == i
        counts[i] = mask.sum()
        if counts[i] > 0:
            correct_counts[i] = correct[mask].sum()

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracy_per_bin = np.where(counts > 0, correct_counts / counts * 100, np.nan)

    return {
        "bin_centers": bin_centers,
        "bin_edges": bin_edges,
        "bin_indices": bin_indices,
        "accuracy": accuracy_per_bin,
        "counts": counts,
        "correct_counts": correct_counts,
        "n_bins": actual_n_bins,
    }


def metrics_table(scores):
    metrics_df = pd.DataFrame(scores).set_index("fold")
    tuple_cols = [c for c in metrics_df.columns if isinstance(c, tuple)]

    metrics_df = metrics_df[tuple_cols]
    metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns)

    summary = pd.concat(
        [
            metrics_df,
            metrics_df.mean().rename("mean").to_frame().T,
            metrics_df.std().rename("std").to_frame().T,
        ]
    )

    row = pd.Series(
        {col: None for col in summary.columns} | {("val", "acc"): 0.5189}, name="baseline"
    )
    summary = pd.concat([summary, row.to_frame().T])

    return summary.map(lambda x: f"{x:.2%}" if isinstance(x, float) else "")


def plot_accuracy_by_bin(bin_data, ax=None, title="Accuracy per predicted return bin"):

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bin_data["bin_centers"], bin_data["accuracy"], linewidth=1.5, marker="o", markersize=3)
    ax.axhline(50, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_ylabel(
        "Accuracy (%)"
        if "(%" in str(ax.get_ylabel()) or "Accuracy" in str(ax.get_ylabel())
        else "Accuracy"
    )
    ax.set_xlabel("Predicted return")
    ax.set_title(title)
    return ax


def print_accuracy_table(bin_data, y_pred):
    print(f"{'Bin':>6} {'pred range':>28} {'Count':>8} {'Accuracy':>10}")
    print("-" * 60)
    low_conf_bins = []
    for i in range(bin_data["n_bins"]):
        acc = bin_data["accuracy"][i]
        if np.isnan(acc):
            continue
        pred_range = f"{bin_data['bin_edges'][i]:.2e} – {bin_data['bin_edges'][i + 1]:.2e}"
        print(f"{i + 1:>6} {pred_range:>28} {bin_data['counts'][i]:>8} {acc:>9.2f}%")
        if acc < 50:
            low_conf_bins.append(i)

    if low_conf_bins:
        max_pred_in_bad = max(
            np.abs(y_pred[bin_data["bin_indices"] == i]).max() for i in low_conf_bins
        )
        low_conf_threshold = max_pred_in_bad
        n_low = sum(bin_data["counts"][i] for i in low_conf_bins)
        print(
            f"\nLow-confidence threshold (|pred| < {low_conf_threshold:.2e}): "
            f"{n_low} rows ({n_low / len(y_pred):.1f}%) "
            f"across bins {', '.join(str(i + 1) for i in low_conf_bins)}"
        )
    else:
        print("\nNo bins with accuracy below 50%.")


def plot_threshold_optimization(y_true, y_pred, q_range=(30, 71, 2)):

    thresholds = np.percentile(y_pred, q=np.arange(*q_range))
    accs = [
        balanced_accuracy_score((y_true > 0).astype(int), (y_pred > t).astype(int))
        for t in thresholds
    ]
    best_t = thresholds[np.argmax(accs)]
    best_acc = max(accs)

    default_idx = np.argmin(np.abs(thresholds))
    default_acc = accs[default_idx]

    _, ax = plt.subplots(figsize=(6, 3))
    ax.plot(thresholds, accs, marker="o")
    ax.axvline(0, color="gray", ls="--", label="default (0)")
    ax.axvline(best_t, color="red", ls="--", label=f"best: {best_t:.2e}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"Optimal threshold tuning (best acc={best_acc * 100:.2f}%)")
    ax.legend()
    plt.show(block=False)

    print(f"Default threshold (0) acc: {default_acc * 100:.2f}%")
    print(f"Optimal threshold ({best_t:.2e}) acc: {best_acc * 100:.2f}%")

    return best_t, best_acc
