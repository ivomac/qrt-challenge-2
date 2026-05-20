# [QRT Challenge](https://challengedata.ens.fr/participants/challenges/167/)

This is mostly a notebook dump of the state of my code at challenge end. Expect some inconsistencies and outdated notebooks. Full pipeline: raw data preprocessing, analysis, feature engineering, LightGBM model fitting.

Best public score I achieved was 52.08%, at rank 245 out of 1180.

Full challenge description in [CHALLENGE.md](CHALLENGE.md).

## Recommended installation

Using a local virtual environment and install the requirements:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A first-time will need to register the env as a kernel to use it in JupyterLab
```
python -m ipykernel install --user --name=qrt-venv --display-name="QRT-venv"
```

## Problem

Each row in the dataset contains historical data on the performance of several portfolios: 20 days of historical returns, volumes, and median turnover.

The goal is to predict whether the next-day return will be positive/negative.

## Data Overview

- 527,073 training observations, 31,870 test observations
- Evaluation metric: sign accuracy (balanced accuracy)

Each row has a specific day tag, allocation/portfolio tag, and group tag. The allocation is a subdivision of group. Each day contains a single row per allocation: *(day, alloc)* tuples are unique.

There are 4 groups, 278 different allocations, and 2522 days in the training set.

Crucially, the days are anonymized and shuffled. The day tags only separate days and carry no order information. The only temporal connection we have is within-row (20 day history) and between rows of the same day.

As far as I could tell from analysis, the 20-day windows of each "day" tag do not overlap: I could not find any shifted correspondence of the returns between rows of the same allocation on different days.

## Benchmarks

A benchmark included with the challenge sets a minimum accuracy of 50.79%.

Data analysis reveals that the last return (`RET_1`) is the most predictive feature of tomorrow's return (I call it `RET_0`): On the training dataset, the sign of `RET_1` agrees with the target `RET_0` 51.89% overall on train.

## Script/Notebook Pipeline

```
data/0-raw/ (CSV)
    v
[1a] Preprocessing
    v
[2a-2e] Analysis (insights for  engineering)
    v
[3a] Post-processing
    v
[3b] Feature Engineering
    v
[3c] Feature Filtering
    v
[4b] LightGBM Huber (pruning) -> submission
[4c] LightGBM Huber (Optuna) -> submission
```

## Quick Start

All scripts are Jupytext percent-format Python files in `notebooks/`. They can be:

- Opened as Jupyter notebooks (`bin/nb push script.py`)
- Run as standalone Python scripts (`python script.py`)

## Pipeline Stages

### Stage 1: Preprocessing

**`1a-data-preprocessing.py`** -- Loads raw CSV data, renames columns (`MEDIAN_DAILY_TURNOVER` to `TURN`, `SIGNED_VOLUME_*` to `SVOL_*`, `ALLOCATION` to `ALLOC`), strips string prefixes from identifiers, merges target into feature DataFrame as `RET_0`, and saves as Parquet.

### Stage 2: Analysis

**`2a-analysis-data_overview.py`** -- Comprehensive data survey covering:

- **Data structure**: SVOL_1 has 73.5% NaN rate (ALLOC-dependent; 24 ALLOCs 0% NaN, 254 ALLOCs >50% NaN). ALLOC 14 and 46 have only 19 train rows each (46 also has duplicate rows). 2,522 unique TS, 278 ALLOCs, 4 GROUPs. RET_0 is 100% NaN in test by design.
- **Feature distributions**: RET_* follow approximate non-central t (clip to +/-0.01 trims 1% tails). SVOL_* are bimodal with peaks at +/-1 and heavy tails (kurtosis 359). TURN has 3-5 log-normal modes spanning 10^-12 to 10.
- **Train vs test**: No TS overlap. All 278 ALLOCs appear in both sets. Train row counts per TS vary widely (19-276); test is uniform (102-116 per ALLOC). GROUP 3 underrepresented in test (-7 pp), GROUP 1 overrepresented (+6 pp).
- **Distribution shifts**: KS tests find statistically significant shifts in 35/41 features, but practically small: post-clip mean |D_pp| = 0.026 (2.6% of a standard deviation). Clipping does not reduce KS because the shift is in the bulk, not tails.
- **Target bias**: Per-allocation target deviations range -3.3% to +12.5% (EB-shrunk). In-sample Spearman rho=+0.74, but out-of-sample drops to rho=+0.28 -- ALLOC aggregates may not generalize.
- **Clip bounds** determined via Q-Q plots: RET +/-0.01, SVOL +/-7.5, TURN [1e-4, 1.6].

**`2b-analysis-time_reverse_eng.py`** -- Attempts to reconstruct the anonymized temporal order by matching overlapping RET sequences. No matches found, time order appears unreconstructable, no data overlap between days. If so, then data augmentation through shifting should be valuable?

**`2c-analysis-signal_and_noise.py`** -- Variance decomposition of the target: 93% is residual noise after removing day and allocation effects. Day variance is approx ~8% and allocation variance is ~1.8%.

**`2d-analysis-correlation.py`** -- Spearman correlations and mutual information with target. RET_1 dominates (rho=+0.061), TURNOVER follows, then RET_7-9 (~1 week ago).

**`2e-analysis-ret1_heuristic.py`** -- The RET\_1 sign baseline achieves 51.89% balanced accuracy. Signal is consistent across 59% of days. Sign accuracy is correlated with RET\_1 magnitude: RET\_0 is more likely to have the same sign as RET\_1 the bigger RET\_1 is. Threshold optimizations (global, per-ALLOC, per-TS) are either weak or do not generalize.

### Stage 3: Features

**`3a-feature-processing.py`** -- Drops ALLOC 14 and 46, applies clip bounds from
2a analysis, normalizes by clip half-range (values ~[-1, 1]), downcasts to float32.

**`3b-feature-engineering.py`** -- Generates ~4800 engineered features using Polars
for memory efficiency:
- Element-wise derived series (VOLUME, pos/neg clips, cross-products, polynomials)
- Timestamp-level statistics (mean, std, max, min, percentile rank)
- TS-demeaned and TS-zscored versions of all base series
- Row aggregates across 5 time windows x 15 vector families x ~10 operations
- Cross-period ratios of row aggregates
- Fold-safe per-allocation statistics (computed from training fold only)
- 12-fold TS-based KFold split with column-pruned

**`3c-feature-filter.py`** -- Computes Spearman rank correlation of all features vs
target, then greedily removes redundant features (|rho| >= 0.95 with any kept
feature) with the same feature mask applied to all folds.

### Stage 4: Fitting

**`4b-fitting-lightgbm_huber.py`** -- LightGBM with Huber loss and iterative
feature selection. Each round trains 12 folds, computes gain importance, and drops
features with low importance (mean/std ratio < threshold). Converges in one round.
Generates test submission via median of fold-level predictions.

**`4c-fitting-lightgbm_huber_optuna.py`** -- Optuna hyperparameter tuning with
200 trials, each running 12-fold CV. Tunes learning rate, tree structure, and
regularization parameters. Uses a curated subset of features. Trial history saved to SQLite.

## Shift Augmentation

The `notebooks/shift_augmented/` directory contains an experiment with time-shift
data augmentation -- creating synthetic training examples by shifting the 20-day
feature window backwards. This approach achieves too high train/val accuracies, it must be leaky but I didn't understand how exactly if the 20-day windows don't have overlaps.

## Project Structure

```
|-- README.md
|-- CHALLENGE.md                 # Full problem description
|-- AGENTS.md
|-- bin/nb                       # Jupytext script
|-- data/
|   |-- 0-raw/                   # Input CSV files
|   |-- ...                      # Staged outputs
|-- notebooks/
    |-- tools.py                 # Shared utilities (I/O, metrics, plotting)
    |-- ...
    |-- archive/                 # Archived experiments
    |   |-- ...
    |-- shift_augmented/         # Shift augmentation experiment
        |-- ...
```

## Notebook Usage

Each `.py` file is a Jupytext percent-format mirror of a Jupyter notebook. Use the
`bin/nb` tool to sync:

```bash
bin/nb pull notebooks/notebook.ipynb   # .ipynb -> .py
bin/nb push notebooks/notebook.ipynb   # .py -> .ipynb
bin/nb run  notebooks/notebook.ipynb   # execute notebook
```

