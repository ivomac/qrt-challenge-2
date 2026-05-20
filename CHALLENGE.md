Challenge context
Trust or Short? Predicting the Performance of daily Asset Allocations

In the world of systematic trading, asset allocations are everywhere — but signal quality is everything.

Each day, traders are flooded with candidate allocations: portfolio constructions based on recent signals, liquidity flows, or historical patterns. Some of these portfolios will perform well in the next trading session. Others will underperform—or worse, underperform so consistently that shorting them might be the more profitable move.

This challenge centers around a simple yet high-stakes question:
Can you predict whether a given asset allocation is worth following — or shorting?
What is an Asset Allocation ?

An asset allocation, is a systematic method of constructing a portfolio of assets using predefined signals or rules. In this challenge, each allocation is defined by a set of portfolio weights, that can be positive or negative, applied on a specific day and held for one trading session. From one day to another, an allocation can rebalance its weights to a certain proportion, called turnover, depending on the rules used to compute it. The returns of an asset allocation reflects the aggregated performance of these weighted positions rebalanced every day.
Mathematical Definition

For a given trading day t, an allocation S, and M assets in a trading universe. Let :

The weights of allocation S at time t :

wS,t=(wS,t,1,wS,t,2,...,wS,t,M)wS,t=(wS,t,1,wS,t,2,...,wS,t,M)

The performance, also often referred as Return, of asset i from day t to day t+1 :

ri,t+1ri,t+1

Then the realized return of an allocation S at t+1 is given by :

rS,t+1=∑i=1MwS,t,i∗ri,t+1rS,t+1=∑i=1MwS,t,i∗ri,t+1
Challenge goals

Each row in the dataset represents a day and an asset allocation, materialized as a portfolio constructed and rebalanced on that day. We give you a history of how that allocation behaved when rebalanced over the past 20 trading days: its daily performance, liquidity behaviour (proxied through weighted volumes), and median turnover.

The goal is to use that historical footprint to predict the sign of the allocation’s performance on the following day.

    If the model predicts positive return → trust the allocation.
    If the model predicts negative return → short the allocation.

Evaluation Metric

You will be evaluated based on accuracy, which measures how often the model correctly predicts the direction (sign) of an allocation’s next-day return.

For each row i indexed by a time stamp t and an allocation S, we provide the true returns r_i for the following trading day. Your model must predict the sign of that return:

    1 if you believe the allocation will have a positive return (go long)
    0 if you believe the return will be negative.

Only the sign is evaluated — not your capacity to predict the return’s magnitude.

Accuracy=1N∑i=1N1[sign(r^i)=sign(ri)]Accuracy=N1∑i=1N1[sign(r^i)=sign(ri)]

Accuracy=1T∗M∑t=1T∑S=1M1[sign(r^S,t+1)=sign(rS,t+1)]Accuracy=T∗M1∑t=1T∑S=1M1[sign(r^S,t+1)=sign(rS,t+1)]

Where:

    N is the number of rows

    sign(x) = 1 if x > 0, else 0

    1 is the indicator function (equals 1 if the condition is true, 0 otherwise)

    The true next day return for allocation S at timestamp t:

ri=rS,t+1ri=rS,t+1

    The predicted next day return for allocation S at timestamp t:

ri^=r^S,t+1ri^=r^S,t+1
Data description

The dataset is formatted as a time series with a multi-index of (date, allocation).
Each row contains:

    20-day history of allocation returns
    20-day history of volume-weighted liquidity behavior
    Allocation median turnover
    Allocation anonymized GROUP
    Next day allocation return: The performance of the allocation on the next day. You are given the true performance for training, but you will be only evaluated on your capacity to predict its sign.

Columns

    TS Timestamp of the snapshot (Dates were anonymized and shuffled so there is no guarantee of continuity even if the labels are called DATE_0001, DATE_0002, DATE_0003 )
    ALLOCATION Name of the Allocation ( ALLOCATION_01 is the same for DATE_0001, DATE_0002 etc…)
    RET_{i} for i in 1,…,20 - Allocation’s return on last day i
    SIGNED_VOLUME_{i} for i in 1,…,20 - Allocation’s signed volume on last day i. See below for definition.
    MEDIAN_DAILY_TURNOVER - Allocation’s median daily turnover. See below for definition.
    GROUP - Anonymized Allocation group.
    TARGET - Allocation’s true next day return.

More details about volumes, allocation weights, and turnover.

At every day t, each allocation S follows this property, given a universe of M trading instruments :

∀t , ∀S:∑i=1M∣wS,i,t∣=1∀t , ∀S:∑i=1M∣wS,i,t∣=1

The SIGNED_VOLUME of an allocation S at t is given by :

VS,t=∑i=1MwS,t,i∗Vi,tVS,t=∑i=1MwS,t,i∗Vi,t

Where V_{i,t} is the total volume traded on the market of asset i during the trading session at timestamp t.
For homogeneity, these V_{S,t} were rescaled in a rolling fashion to ensure comparability across different style of allocations

The MEDIAN_DAILY_TURNOVER of an allocation S at t is given by :

TOS,t=TURNOVERS,t=∑i=1M∣wS,t,i−wS,t−1,i∣TOS,t=TURNOVERS,t=∑i=1M∣wS,t,i−wS,t−1,i∣

MDTS,t=median(TOS,t,TOS,t−1,...,TOS,t−20)MDTS,t=median(TOS,t,TOS,t−1,...,TOS,t−20)
Files

All files are indexed by a unique ROW_ID, refering to a unique tuple (date, allocation), allowing you to map X_train with Y_train

    X_train.csv - the training set features
    y_train.csv - the training set target
    X_test.csv - the test set features
    sample_submission.csv - a random submission file in the correct format
    benchmark_submission.ipynb - a benchmark submission notebook to generate the benchmark you see in the leaderboard.

527073 observations (i.e. lines) are available in the train set, while 31870 observations are in the test set.
Benchmark description

You will find in the data section a Benchmark Notebook generating the benchmark submission you see in the leaderboard.

The Benchmark Notebook also allows you to see how to correctly generate a submission file.

Here is a breakdown of what you can find in the notebook :

    additional features representing the average historical performance of the allocations on multiple windows
    additional features representing the average historical performance of all allocations on multiple windows
    additional features representing the historical volatility of each allocation on the past 20 days
    additional features representing the average historical volatility of all allocation on the past 20 days
    a ridge fitted on all + additional features.
    a lightgbm model fitted on all + additional features, with a cross validation section ( benchmark submitted ).

The public score, visible on the leaderboard, of the lgbm model is 0.5079.


