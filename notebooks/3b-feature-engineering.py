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
# # Feature Engineering
#
# **Purpose**: Generate derived features from base (clipped, normalized) features.
#
# **Key insights that influence later decisions:**
# - RET_1 dominates feature importance (ρ=0.06). TURN is second.
# - Day-level effects explain 8.3% of target variance — regime features matter.
# - SVOL features have significant train→test distribution shift.
# - Bias features MUST use RET_* sign as signal, not target labels, so they
#   generalize to test TS values (which are not in any training fold).
#
# **Transformation pipeline:**
#
# 1. **E1** — element-wise derived series (VOLUME, pos/neg clips, cross-products/ratios)
# 2. **D1** — TS stats (mean, std, max, min, rank) for base series
# 3. **E2** — TS broadcast: ts_demean and ts_zscore series for all base families
# 4. **R1** — row aggregates: all families × all periods × all ops
# 5. **D2** — TS stats of selected row aggregates (mean, std, max, min, rank)
# 6. **E3** — broadcast D2 back: diff and ratio of R1 vs its D2 mean
# 7. **Alloc** — allocation-level aggs (fold-safe): means, sharpe, diff/ratio, sign dev/bias
# 8. **D1-ts** — TS-level sign dev/bias for key RET columns (computed in D1 alongside ts_mean/std)
#
# **Naming convention:**
# - `{scope}_{op}_{transform?}_{base}_{window?}` for row/ts features
# - `{op}_{a}_{b}` for element-wise combinations of two columns
# - `{scope}_{op}_{base}` for bias features, e.g. `alloc_dev_RET_1`
#
# NaN handling: row aggregates use nan-safe functions (nanmean, nanstd, etc.).
# All-NaN windows produce 0.0 for aggregate features.
# LightGBM handles NaN natively — no filling or flagging needed.

# %%
import tools

import gc
import shutil
import time

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import scipy.stats as st
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


_SCRIPT_T0 = time.perf_counter()


def diag(label: str, df: pl.DataFrame = None) -> None:
    """Print elapsed time, free RAM, free disk, and optional DataFrame shape/size."""
    for disk_path in ("/mnt/SSD", "/home", "/"):
        try:
            disk_free_gb = shutil.disk_usage(disk_path).free / 1e9
            break
        except (FileNotFoundError, PermissionError):
            disk_free_gb = 0.0

    avail_ram_mb = 0.0
    used_ram_mb = 0.0
    ram_path = Path("/proc/meminfo")
    meminfo = {}
    for line in ram_path.read_text().splitlines():
        parts_l = line.split()
        if len(parts_l) >= 2:
            meminfo[parts_l[0].rstrip(":")] = int(parts_l[1])
    avail_ram_mb = meminfo.get("MemAvailable", 0) / 1024
    total_ram_mb = meminfo.get("MemTotal", 0) / 1024
    used_ram_mb = total_ram_mb - avail_ram_mb

    elapsed = time.perf_counter() - _SCRIPT_T0
    parts = [
        f"[{label}]",
        f"t={elapsed:.0f}s",
        f"RAM used/avail: {used_ram_mb:.0f}/{avail_ram_mb:.0f} MB",
        f"Disk free: {disk_free_gb:.1f} GB",
    ]
    if df is not None:
        parts.append(f"df: {df.shape}, ~{df.estimated_size() / 1e9:.2f} GB")
    print("  ".join(parts), flush=True)


# %%
N_SPLITS = 12
PARQUET_ROW_GROUP_SIZE = (
    10_000  # limits dirty-page spike during write; None = one giant group (OOM risk)
)
LAGS = 20
PERIODS = [
    (1, 4),  # short recent
    (1, 7),  # week
    (7, 20),  # one-week-ago till-end
    (1, 14),  # recent two weeks
    (1, 20),  # full history
]

VECTOR_FAMILIES = [
    "RET",
    "SVOL",
    "VOLUME",
    "RET_pos",
    "RET_neg",
    "SVOL_pos",
    "SVOL_neg",
    "prod_RET_SVOL",
    "prod_RET_VOLUME",
    "ratio_RET_VOLUME",
    "ratio_RET_SVOL",
    "sq_RET",
    "cube_RET",
    "ssqrt_RET",
    "sq_SVOL",
    "cube_SVOL",
    "ssqrt_SVOL",
    "SVOL_is_plus1",
    "ts_zscore_RET",
    "ts_zscore_SVOL",
    "ts_pct_rank_RET",
    "ts_pct_rank_SVOL",
]

D2_FAMILIES = ["RET", "SVOL", "VOLUME"]
D2_ROW_OPS = ["mean", "std", "sharpe"]
TURN_SCALARS = ["TURN", "ts_mean_TURN", "ts_std_TURN"]

# Columns for which sign dev/bias is computed at both TS and alloc level.
# RET_1 is a base col (available in D1); row aggregates are available after R1.
BIAS_SRCS = ["RET_1"] + [
    f"row_{op}_RET_{begin}_{end}" for op in ("mean", "sharpe") for begin, end in PERIODS
]


def compute_sign_bias(df: pl.DataFrame, group_col: str, col: str) -> pl.DataFrame:
    """Return mapping DataFrame with group_col + dev/bias for joining.

    dev = positive_rate - 0.5
    bias = sign(dev) * |log10(binomtest p-value)|
    """
    agg = df.group_by(group_col).agg(
        (pl.col(col).drop_nulls() > 0).sum().alias("pos"),
        pl.col(col).drop_nulls().count().alias("total"),
    )

    keys = agg[group_col].to_list()
    pos_vals = agg["pos"].to_list()
    total_vals = agg["total"].to_list()

    dev_vals = []
    bias_vals = []
    for n_pos, n_tot in zip(pos_vals, total_vals):
        if n_tot <= 1:
            dev_vals.append(0.0)
            bias_vals.append(0.0)
        else:
            d = n_pos / n_tot - 0.5
            dev_vals.append(d)
            p = st.binomtest(n_pos, n_tot, p=0.5).pvalue
            bias_vals.append(np.sign(d) * np.abs(np.log10(max(float(p), 1e-300))))

    return pl.DataFrame(
        {
            group_col: keys,
            "dev": dev_vals,
            "bias": bias_vals,
        }
    )


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Generate derived features. Uses float32 throughout to halve memory vs float64.

    Operates column-by-column for groupby transforms to avoid materializing
    large intermediate DataFrames. Row aggregate math still uses numpy arrays
    extracted from polars columns.
    """
    f32 = np.float32

    ret_cols = [f"RET_{i}" for i in range(1, 21)]
    sv_cols = [f"SVOL_{i}" for i in range(1, 21)]

    rets = df.select(ret_cols).to_numpy().astype(f32)
    svols = df.select(sv_cols).to_numpy().astype(f32)

    # --- E1: element-wise derived series ---
    for i in range(1, 21):
        ret = rets[:, i - 1]
        sv = svols[:, i - 1]
        vol = np.abs(sv)

        new_cols = []
        new_cols.append(pl.Series(f"VOLUME_{i}", vol))
        new_cols.append(pl.Series(f"RET_pos_{i}", np.where(ret > 0, ret, f32(0.0)).astype(f32)))
        new_cols.append(pl.Series(f"RET_neg_{i}", np.where(ret < 0, ret, f32(0.0)).astype(f32)))
        new_cols.append(pl.Series(f"SVOL_pos_{i}", np.where(sv > 0, sv, f32(0.0)).astype(f32)))
        new_cols.append(pl.Series(f"SVOL_neg_{i}", np.where(sv < 0, sv, f32(0.0)).astype(f32)))
        new_cols.append(pl.Series(f"prod_RET_SVOL_{i}", (ret * sv).astype(f32)))
        new_cols.append(pl.Series(f"prod_RET_VOLUME_{i}", (ret * vol).astype(f32)))

        out = np.full(len(ret), np.nan, dtype=f32)
        new_cols.append(
            pl.Series(f"ratio_RET_VOLUME_{i}", np.divide(ret, vol, out=out, where=vol != 0.0))
        )
        out = np.full(len(ret), np.nan, dtype=f32)
        new_cols.append(
            pl.Series(f"ratio_RET_SVOL_{i}", np.divide(ret, sv, out=out, where=sv != 0.0))
        )

        new_cols.append(pl.Series(f"sq_RET_{i}", (ret**2).astype(f32)))
        new_cols.append(pl.Series(f"cube_RET_{i}", (ret**3).astype(f32)))
        new_cols.append(
            pl.Series(f"ssqrt_RET_{i}", (np.sign(ret) * np.sqrt(np.abs(ret))).astype(f32))
        )
        new_cols.append(pl.Series(f"sq_SVOL_{i}", (sv**2).astype(f32)))
        new_cols.append(pl.Series(f"cube_SVOL_{i}", (sv**3).astype(f32)))
        new_cols.append(
            pl.Series(f"ssqrt_SVOL_{i}", (np.sign(sv) * np.sqrt(np.abs(sv))).astype(f32))
        )

        eps = np.finfo(f32).eps * 10
        is_plus1 = (np.abs(sv - f32(1.0)) < eps).astype(np.uint8)
        is_minus1 = (np.abs(sv - f32(-1.0)) < eps).astype(np.uint8)
        new_cols.append(pl.Series(f"SVOL_is_plus1_{i}", is_plus1))
        new_cols.append(pl.Series(f"SVOL_is_minus1_{i}", is_minus1))
        new_cols.append(pl.Series(f"SVOL_is_peak_{i}", (is_plus1 | is_minus1).astype(np.uint8)))

        df = df.with_columns(new_cols)

    vol_cols = [f"VOLUME_{i}" for i in range(1, 21)]
    del rets, svols
    gc.collect()
    print("  E1 done", flush=True)
    diag("E1", df)

    # --- D1: TS stats for all base series ---
    ts_base = ret_cols + sv_cols + vol_cols + ["TURN"]
    d1_exprs = []
    for col in ts_base:
        d1_exprs.extend(
            [
                pl.col(col).mean().over("TS").alias(f"ts_mean_{col}"),
                pl.col(col).std().over("TS").alias(f"ts_std_{col}"),
                pl.col(col).max().over("TS").alias(f"ts_max_{col}"),
                pl.col(col).min().over("TS").alias(f"ts_min_{col}"),
                (
                    pl.col(col).rank("ordinal").over("TS").cast(pl.Float64)
                    / pl.col(col).count().over("TS")
                )
                .cast(pl.Float32)
                .alias(f"ts_pct_rank_{col}"),
            ]
        )
    df = df.with_columns(d1_exprs)

    mapping = compute_sign_bias(df, "TS", "RET_1")
    df = df.join(
        mapping.select(pl.all().name.prefix("_sb_")),
        left_on="TS",
        right_on="_sb_TS",
        how="left",
    ).rename({"_sb_dev": "ts_dev_RET_1", "_sb_bias": "ts_bias_RET_1"})

    print("  D1 done", flush=True)
    diag("D1", df)

    # --- E2: TS demean + zscore series for all base families ---
    for col in ts_base:
        ts_mean = df[f"ts_mean_{col}"].to_numpy().astype(f32)
        ts_std = df[f"ts_std_{col}"].to_numpy().astype(f32)
        col_vals = df[col].to_numpy().astype(f32)
        demeaned = col_vals - ts_mean
        out = np.full_like(demeaned, np.nan)
        zscored = np.divide(demeaned, ts_std, out=out, where=ts_std != 0)
        df = df.with_columns(
            pl.Series(f"ts_demean_{col}", demeaned),
            pl.Series(f"ts_zscore_{col}", zscored),
        )

    print("  E2 done", flush=True)
    diag("E2", df)

    # --- R1: row aggregates — all families × all periods × all ops (nan-safe) ---
    n = len(df)

    for _fam_idx, family in enumerate(VECTOR_FAMILIES):
        cols = [f"{family}_{i}" for i in range(1, LAGS + 1)]
        mat = df.select(cols).to_numpy().astype(f32)
        lag1 = mat[:, 0]

        for den in TURN_SCALARS:
            denom = np.abs(df[den].to_numpy().astype(f32))
            out = np.full(n, np.nan, dtype=f32)
            df = df.with_columns(
                pl.Series(
                    f"ratio_{cols[0]}_{den}",
                    np.divide(lag1, denom, out=out, where=denom != 0),
                )
            )

        for begin, end in PERIODS:
            p_label = f"{begin}_{end}"
            w = mat[:, begin - 1 : end]
            w_abs = np.abs(w)

            rm = np.nanmean(w, axis=1).astype(f32)
            rs = np.nanstd(w, axis=1).astype(f32)

            col_max = np.nanmax(w, axis=1).astype(f32)
            col_min = np.nanmin(w, axis=1).astype(f32)

            out_sharpe = np.full(n, np.nan, dtype=f32)

            n_lags = end - begin + 1
            weights = np.linspace(1, 0.1, n_lags, dtype=f32)
            wsum = np.nansum(w * weights, axis=1).astype(f32)

            tmp_cube = np.nanmean(w**3, axis=1).astype(f32)
            tmp_sqrt = np.nanmean(np.sqrt(w_abs), axis=1).astype(f32)
            tmp_log = np.nanmean(np.log(w_abs + f32(1e-20)), axis=1).astype(f32)
            tmp_inv = np.nanmean(1.0 / (w_abs + f32(1e-20)), axis=1).astype(f32)

            # Rank of the first element of the window among all window elements.
            # count-less is 10x faster than double argsort for a single-element rank.
            sub_mat = w.astype(np.float64)
            nan_mask = np.isnan(sub_mat)
            target = sub_mat[:, 0:1]
            valid_counts = np.sum(~nan_mask, axis=1, dtype=np.float64)
            less_count = np.sum((sub_mat < target) & ~nan_mask, axis=1, dtype=np.float64)
            row_pct_val = np.where(valid_counts > 1, less_count / (valid_counts - 1), 0.0).astype(
                f32
            )
            row_pct_val[nan_mask[:, 0]] = np.nan
            rank_cols = cols[begin - 1 : end]

            diff_val = (lag1 - rm).astype(f32)
            out_ratio = np.full(n, np.nan, dtype=f32)

            batch_cols = [
                pl.Series(f"row_mean_{family}_{p_label}", rm),
                pl.Series(f"row_std_{family}_{p_label}", rs),
                pl.Series(f"row_max_{family}_{p_label}", col_max),
                pl.Series(f"row_min_{family}_{p_label}", col_min),
                pl.Series(
                    f"row_sharpe_{family}_{p_label}",
                    np.divide(rm, rs, out=out_sharpe, where=rs > 0),
                ),
                pl.Series(f"row_weighted_sum_{family}_{p_label}", wsum),
                pl.Series(f"row_mean_cube_{family}_{p_label}", tmp_cube),
                pl.Series(f"row_mean_sqrt_{family}_{p_label}", tmp_sqrt),
                pl.Series(f"row_mean_log_{family}_{p_label}", tmp_log),
                pl.Series(f"row_mean_inv_{family}_{p_label}", tmp_inv),
                pl.Series(f"row_pct_rank_{rank_cols[0]}_{p_label}", row_pct_val),
                pl.Series(f"diff_{cols[0]}_row_mean_{family}_{p_label}", diff_val),
                pl.Series(
                    f"ratio_{cols[0]}_row_mean_{family}_{p_label}",
                    np.divide(lag1, np.abs(rm), out=out_ratio, where=rm != 0),
                ),
            ]

            for den in TURN_SCALARS:
                denom = np.abs(df[den].to_numpy().astype(f32))
                out_rmean = np.full(n, np.nan, dtype=f32)
                out_rstd = np.full(n, np.nan, dtype=f32)
                batch_cols.append(
                    pl.Series(
                        f"ratio_row_mean_{family}_{p_label}_{den}",
                        np.divide(rm, denom, out=out_rmean, where=denom != 0),
                    )
                )
                batch_cols.append(
                    pl.Series(
                        f"ratio_row_std_{family}_{p_label}_{den}",
                        np.divide(rs, denom, out=out_rstd, where=denom != 0),
                    )
                )

            df = df.with_columns(batch_cols)

        del mat
        if _fam_idx % 5 == 4:
            diag(f"R1-fam{_fam_idx + 1}/{len(VECTOR_FAMILIES)}", df)

    print("  R1 done", flush=True)
    diag("R1", df)

    # --- R2: cross-period ratios of row aggregates ---
    print("  R2: cross-period ratios...", flush=True)
    r2_cols = []
    for family in D2_FAMILIES:
        for row_op in D2_ROW_OPS:
            period_srcs = [f"row_{row_op}_{family}_{begin}_{end}" for begin, end in PERIODS]
            for i, src_a in enumerate(period_srcs):
                a_vals = df[src_a].to_numpy().astype(f32)
                for src_b in period_srcs[i + 1 :]:
                    b_vals = df[src_b].to_numpy().astype(f32)
                    out = np.full(n, np.nan, dtype=f32)
                    r2_cols.append(
                        pl.Series(
                            f"ratio_{src_a}_{src_b}",
                            np.divide(a_vals, np.abs(b_vals), out=out, where=b_vals != 0),
                        )
                    )
    df = df.with_columns(r2_cols)
    diag("R2", df)
    print("  R2 done", flush=True)

    # --- B2: TS-level sign bias for row aggregates ---
    print("  B2: TS bias for row aggregates...", flush=True)
    for col in BIAS_SRCS[1:]:  # RET_1 already done in D1
        mapping = compute_sign_bias(df, "TS", col)
        df = df.join(
            mapping.select(pl.all().name.prefix("_sb_")),
            left_on="TS",
            right_on="_sb_TS",
            how="left",
        ).rename({"_sb_dev": f"ts_dev_{col}", "_sb_bias": f"ts_bias_{col}"})
    print("  B2 done", flush=True)

    # --- D2: TS stats of selected row aggregates ---
    # --- E3: broadcast D2 mean back via diff and ratio ---
    gc.collect()
    diag("pre-D2", df)

    # Pass 1: batch all over() expressions in a single with_columns call.
    d2_srcs = [
        f"row_{row_op}_{family}_{begin}_{end}"
        for family in D2_FAMILIES
        for row_op in D2_ROW_OPS
        for begin, end in PERIODS
    ]
    print(
        f"  D2: computing stats for {len(d2_srcs)} sources ({len(d2_srcs) * 5} expressions)...",
        flush=True,
    )
    d2_exprs = []
    for src in d2_srcs:
        d2_exprs.extend(
            [
                pl.col(src).mean().over("TS").alias(f"ts_mean_{src}"),
                pl.col(src).std().over("TS").alias(f"ts_std_{src}"),
                pl.col(src).max().over("TS").alias(f"ts_max_{src}"),
                pl.col(src).min().over("TS").alias(f"ts_min_{src}"),
                (
                    pl.col(src).rank("ordinal").over("TS").cast(pl.Float64)
                    / pl.col(src).count().over("TS")
                )
                .cast(pl.Float32)
                .alias(f"ts_pct_rank_{src}"),
            ]
        )
    df = df.with_columns(d2_exprs)
    diag("post-D2", df)

    # Pass 2: numpy diff/ratio (reads ts_mean_* now present in df).
    print("  E3: computing diff/ratio...", flush=True)
    e3_cols = []
    for src in d2_srcs:
        src_vals = df[src].to_numpy().astype(f32)
        ts_mean_vals = df[f"ts_mean_{src}"].to_numpy().astype(f32)
        out = np.full(len(src_vals), np.nan, dtype=f32)
        e3_cols.extend(
            [
                pl.Series(f"diff_{src}_ts_mean_{src}", (src_vals - ts_mean_vals).astype(f32)),
                pl.Series(
                    f"ratio_{src}_ts_mean_{src}",
                    np.divide(src_vals, np.abs(ts_mean_vals), out=out, where=ts_mean_vals != 0),
                ),
            ]
        )
    df = df.with_columns(e3_cols)

    diag("post-E3", df)
    print("  D2+E3 done", flush=True)

    return df


# %%
def _alloc_src_list() -> list[str]:
    sources = []
    for family in D2_FAMILIES:
        for row_op in D2_ROW_OPS:
            for begin, end in PERIODS:
                sources.append(f"row_{row_op}_{family}_{begin}_{end}")
    return sources


def _compute_alloc_stats(src_path: Path, src_list: list[str]) -> tuple[pl.DataFrame, dict]:
    """Compute alloc stats via column-pruned lazy scan — never loads all 4801 columns.

    Reads only ALLOC + src_list + TURN + RET bias cols (~64 cols vs 4801).
    """
    sel_cols = ["ALLOC"] + src_list + ["TURN"]
    lf = pl.scan_parquet(src_path).select(sel_cols)

    agg_exprs = []
    for src in src_list:
        agg_exprs.append(pl.col(src).mean().alias(f"alloc_mean_{src}"))
        agg_exprs.append(pl.col(src).std().alias(f"alloc_std_{src}"))
    agg_exprs += [
        pl.col("TURN").mean().alias("alloc_mean_TURN"),
        pl.col("TURN").std().alias("alloc_std_TURN"),
        pl.col("TURN").max().alias("alloc_max_TURN"),
        pl.col("TURN").min().alias("alloc_min_TURN"),
    ]
    alloc_stats = lf.group_by("ALLOC").agg(agg_exprs).collect()
    # polars returns Float64 for mean/std of Float32 columns; downcast.
    stat_cols = [c for c in alloc_stats.columns if c != "ALLOC"]
    alloc_stats = alloc_stats.with_columns([pl.col(c).cast(pl.Float32) for c in stat_cols])

    bias_lf = pl.scan_parquet(src_path).select(["ALLOC"] + BIAS_SRCS)
    bias_agg_exprs = []
    for col in BIAS_SRCS:
        bias_agg_exprs.append((pl.col(col).drop_nulls() > 0).sum().alias(f"_bp_{col}"))
        bias_agg_exprs.append(pl.col(col).drop_nulls().count().alias(f"_bt_{col}"))
    bias_agg = bias_lf.group_by("ALLOC").agg(bias_agg_exprs).collect()

    bias_keys = bias_agg["ALLOC"].to_list()
    bias_maps: dict[str, dict] = {}
    for col in BIAS_SRCS:
        pos_vals = bias_agg[f"_bp_{col}"].to_list()
        total_vals = bias_agg[f"_bt_{col}"].to_list()
        dev_vals, bias_vals = [], []
        for n_pos, n_tot in zip(pos_vals, total_vals):
            if n_tot <= 1:
                dev_vals.append(0.0)
                bias_vals.append(0.0)
            else:
                d = n_pos / n_tot - 0.5
                dev_vals.append(d)
                p = st.binomtest(n_pos, n_tot, p=0.5).pvalue
                bias_vals.append(np.sign(d) * np.abs(np.log10(max(float(p), 1e-300))))
        bias_maps[f"alloc_dev_{col}"] = dict(zip(bias_keys, dev_vals))
        bias_maps[f"alloc_bias_{col}"] = dict(zip(bias_keys, bias_vals))

    return alloc_stats, bias_maps


def _enrich_and_sink(
    src_path: Path,
    alloc_stats: pl.DataFrame,
    bias_maps: dict[str, dict],
    dst_path: Path,
    src_list: list[str],
) -> None:
    """Stream-enrich src_path with alloc features and sink to dst_path.

    Uses sink_parquet so the enriched DataFrame is never fully materialized —
    only one row-group batch (~row_group_size rows) lives in memory at a time.
    When src_path == dst_path, writes to a temp file then replaces atomically.
    """
    lf = pl.scan_parquet(src_path)
    lf = lf.join(alloc_stats.lazy(), on="ALLOC", how="left")

    derived_exprs = []
    for src in src_list:
        m = pl.col(f"alloc_mean_{src}")
        s = pl.col(f"alloc_std_{src}")
        v = pl.col(src)
        derived_exprs += [
            pl.when(s.abs() > 0)
            .then(m / s.abs())
            .otherwise(None)
            .cast(pl.Float32)
            .alias(f"alloc_sharpe_{src}"),
            (v - m).cast(pl.Float32).alias(f"diff_{src}_alloc_mean_{src}"),
            pl.when(m != 0)
            .then(v / m.abs())
            .otherwise(None)
            .cast(pl.Float32)
            .alias(f"ratio_{src}_alloc_mean_{src}"),
        ]
    m_t = pl.col("alloc_mean_TURN")
    s_t = pl.col("alloc_std_TURN")
    derived_exprs.append(
        pl.when(s_t.abs() > 0)
        .then(m_t / s_t.abs())
        .otherwise(None)
        .cast(pl.Float32)
        .alias("alloc_sharpe_TURN")
    )
    lf = lf.with_columns(derived_exprs)

    cross_exprs = []
    for src in src_list:
        ts_m = pl.col(f"ts_mean_{src}")
        alloc_m = pl.col(f"alloc_mean_{src}")
        cross_exprs.append(
            pl.when(ts_m != 0)
            .then(alloc_m / ts_m.abs())
            .otherwise(None)
            .cast(pl.Float32)
            .alias(f"ratio_alloc_mean_{src}_ts_mean_{src}")
        )
    lf = lf.with_columns(cross_exprs)

    bias_exprs = []
    for col in BIAS_SRCS:
        for suffix in ("dev", "bias"):
            name = f"alloc_{suffix}_{col}"
            bias_exprs.append(
                pl.col("ALLOC")
                .replace_strict(bias_maps[name], default=None)
                .cast(pl.Float32)
                .alias(name)
            )
    lf = lf.with_columns(bias_exprs)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() == dst_path.resolve():
        tmp = dst_path.with_suffix(".tmp.parquet")
        lf.sink_parquet(tmp, compression="uncompressed", row_group_size=PARQUET_ROW_GROUP_SIZE)
        tmp.replace(dst_path)
    else:
        lf.sink_parquet(dst_path, compression="uncompressed", row_group_size=PARQUET_ROW_GROUP_SIZE)


ID_LIKE = ("ROW_ID", "TS", "ALLOC", "GROUP")


def constant_cols(df: pl.DataFrame, protect: tuple = ID_LIKE) -> list[str]:
    # one streaming null_count/min/max pass is far cheaper than n_unique here
    n = len(df)
    feats = [c for c in df.columns if c not in protect]
    agg = df.select(
        [pl.col(c).null_count().alias(f"_nc_{c}") for c in feats]
        + [pl.col(c).min().alias(f"_mn_{c}") for c in feats]
        + [pl.col(c).max().alias(f"_mx_{c}") for c in feats]
    ).row(0, named=True)
    out = []
    for c in feats:
        if agg[f"_nc_{c}"] == n or agg[f"_mn_{c}"] == agg[f"_mx_{c}"]:
            out.append(c)
    return out


def convert_types_pl(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast float64->float32, int64->int32."""
    casts = []
    for c, dtype in df.schema.items():
        if dtype == pl.Float64:
            casts.append(pl.col(c).cast(pl.Float32))
        elif dtype == pl.Int64:
            casts.append(pl.col(c).cast(pl.Int32))
    if casts:
        df = df.with_columns(casts)
    return df


# %%
diag("start")
tools.clean("3b-engineered")

base = Path("../data/3a-postprocessed/full")
full_train_path = Path("../data/3b-engineered/_full_X_train.parquet")
full_test_path = Path("../data/3b-engineered/_full_X_test.parquet")
fold_dir = Path("../data/3b-engineered/folds")

# --- Phase 1: feature-engineer train alone, save, free ---
train = pl.read_parquet(base / "train.parquet")
print(f"Loaded train {train.shape}")
diag("loaded train", train)

t0 = time.perf_counter()
print("add_features(train)...", flush=True)
train = add_features(train)
print(f"  done: {train.shape}  ({time.perf_counter() - t0:.0f}s)", flush=True)
diag("after add_features(train)", train)

train = convert_types_pl(train)
diag("after convert_types(train)", train)

# computed on train, applied to test too so the column sets match
const_cols = constant_cols(train)
if const_cols:
    train = train.drop(const_cols)
print(f"  dropped {len(const_cols)} constant cols -> {train.shape[1]} cols", flush=True)
diag("after drop_constant(train)", train)

train_dates = train["TS"].unique().to_numpy()
split_ids = list(KFold(n_splits=N_SPLITS, random_state=0, shuffle=True).split(train_dates))

full_train_path.parent.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
train.write_parquet(
    full_train_path, compression="uncompressed", row_group_size=PARQUET_ROW_GROUP_SIZE
)
print(
    f"  saved full train ({full_train_path.stat().st_size / 1e9:.1f} GB, {time.perf_counter() - t0:.0f}s)",
    flush=True,
)
del train
gc.collect()
diag("after save+free train")

# --- Phase 2: feature-engineer test alone, save, free ---
test = pl.read_parquet(base / "test.parquet")
print(f"Loaded test {test.shape}")
t0 = time.perf_counter()
print("add_features(test)...", flush=True)
test = add_features(test)
print(f"  done: {test.shape}  ({time.perf_counter() - t0:.0f}s)", flush=True)
test = convert_types_pl(test)
test = test.drop([c for c in const_cols if c in test.columns])
diag("after add_features(test)", test)

test.write_parquet(
    full_test_path, compression="uncompressed", row_group_size=PARQUET_ROW_GROUP_SIZE
)
del test
gc.collect()
diag("after save+free test")

# --- Phase 3: single-pass fold split ---
# Each TS belongs to 11 train sets + 1 val set (KFold with shuffle=True).
# Old 24-scan approach: 24 reads of the 7.9 GB file, 24 writes.
# New approach: 1 read, 24 simultaneous writes — same ~92 GB total output.
print("Splitting folds (single pass)...", flush=True)
diag("fold split before")

_train_sets: list[set[str]] = []
_val_sets: list[set[str]] = []
for _k, (_tr_idx, _va_idx) in enumerate(split_ids):
    _train_sets.append({str(_ts) for _ts in train_dates[_tr_idx].tolist()})
    _val_sets.append({str(_ts) for _ts in train_dates[_va_idx].tolist()})

_writers: dict[tuple[int, str], pq.ParquetWriter] = {}
_row_counts: dict[tuple[int, str], int] = {}
_pf = pq.ParquetFile(str(full_train_path))
_cols = [c for c in _pf.schema_arrow.names if c != "index"]

_batch_num = 0
_n_batches = _pf.metadata.num_row_groups
for _batch in _pf.iter_batches(batch_size=PARQUET_ROW_GROUP_SIZE, columns=_cols):
    _batch_num += 1
    if _batch_num % 20 == 0:
        diag(f"fold split batch {_batch_num}/{_n_batches}", None)

    _str_ts = [str(_ts) for _ts in _batch.column("TS").to_pylist()]

    for _k in range(N_SPLITS):
        for _sn, _ts_set in [("train", _train_sets[_k]), ("test", _val_sets[_k])]:
            _indices = [i for i, ts in enumerate(_str_ts) if ts in _ts_set]
            if not _indices:
                continue
            _sub = _batch.take(_indices)
            _wkey = (_k, _sn)
            if _wkey not in _writers:
                _out = fold_dir / str(_k) / f"{_sn}.parquet"
                _out.parent.mkdir(parents=True, exist_ok=True)
                _writers[_wkey] = pq.ParquetWriter(str(_out), _sub.schema, compression="none")
            _writers[_wkey].write_table(pa.Table.from_batches([_sub]))
            _row_counts[_wkey] = _row_counts.get(_wkey, 0) + len(_indices)
    del _str_ts

for _w in _writers.values():
    _w.close()
for (_k, _sn), _n in sorted(_row_counts.items()):
    print(f"  fold {_k} {_sn}: {_n} rows", flush=True)
print(f"  total rows written: {sum(_row_counts.values())}", flush=True)
del _writers, _row_counts, _train_sets, _val_sets, _pf
gc.collect()
print("  done", flush=True)
diag("fold split after")

for _k in range(N_SPLITS):
    for _sn in ("train", "test"):
        _p = fold_dir / str(_k) / f"{_sn}.parquet"
        if not _p.exists():
            raise FileNotFoundError(f"missing fold file: {_p}")
print("  all 24 fold files verified", flush=True)

# --- Phase 4: add fold-safe alloc features to each fold ---
# _compute_alloc_stats uses column-pruned lazy scan (~64 cols, not 4801).
# _enrich_and_sink streams the enriched result via sink_parquet — never
# materializes the full enriched DataFrame in RAM.
fold_src_list = _alloc_src_list()

for k in range(N_SPLITS):
    t0 = time.perf_counter()
    print(f"-- Fold {k} --", flush=True)
    diag(f"fold {k} start")

    fold_path = fold_dir / str(k)
    fold_out_dir = Path(f"../data/3b-engineered/folds/{k}")

    alloc_stats, bias_maps = _compute_alloc_stats(fold_path / "train.parquet", fold_src_list)
    diag(f"fold {k} stats computed")

    _enrich_and_sink(
        fold_path / "train.parquet",
        alloc_stats,
        bias_maps,
        fold_out_dir / "train.parquet",
        fold_src_list,
    )
    _enrich_and_sink(
        fold_path / "test.parquet",
        alloc_stats,
        bias_maps,
        fold_out_dir / "test.parquet",
        fold_src_list,
    )

    print(f"  fold {k} total: {time.perf_counter() - t0:.0f}s", flush=True)
    diag(f"fold {k} done")

# --- Phase 5: full-data alloc features ---
# Same lazy streaming approach as Phase 4.
print("-- Full --", flush=True)
diag("before full fold agg")

full_src_list = _alloc_src_list()
full_out_dir = Path("../data/3b-engineered/full")

alloc_stats, bias_maps = _compute_alloc_stats(full_train_path, full_src_list)
diag("full stats computed")

_enrich_and_sink(
    full_train_path, alloc_stats, bias_maps, full_out_dir / "train.parquet", full_src_list
)
_enrich_and_sink(
    full_test_path, alloc_stats, bias_maps, full_out_dir / "test.parquet", full_src_list
)
