# Evaluation protocol (frozen before any model was run)

Status: **FROZEN** at git tag `protocol-frozen` (pushed to GitHub before any
model was trained or any final-test target value was read). Any later change is
recorded in the *Deviations log* at the bottom, with a date and a reason, and is
never edited into the text above it.

## 0. What was and was not looked at before freezing

Looked at: the file's column names, the first two rows, row count, the timestamp
grid (gap-free 10 min, 19,735 rows, Mon 2016-01-11 17:00 to Fri 2016-05-27 18:00),
and, on the **development region only**, the interpolation structure of the
weather columns (section 3). Not looked at: any `Appliances` value after
2016-04-22, any model output, any error metric. The dataset is public and has a
published random-split study (Candanedo et al. 2017); I have not used its results
to set any choice below.

## 1. Research question

> When household conditions drift from winter into spring, which of the usual
> forecasting choices still hold up: the model class, the input groups
> (own history / indoor signals / weather), the retraining policy, and the
> calibration of prediction intervals?

This is an evaluation-design study on one house and 4.5 months. It does not
claim general results about residential energy.

## 2. Task

Forecast `Appliances` (Wh per 10-minute step) at **horizons h in {1, 6, 144}
steps** (10 min, 1 h, 1 day ahead). A forecast is issued at origin row `i` after
row `i` is complete, and targets row `i+h`. Only information from rows `<= i`
(plus calendar features of the target time, which are known in advance) may be
used. The original data paper predicts the *same-time* energy from concurrent
sensors; that is a regression, not a forecast, and is not what is done here.

## 3. Leakage rules

1. Features at origin `i` are computed only from rows `<= i`. Enforced by a unit
   test that corrupts every row after `i` and requires identical features.
2. **Airport-weather columns** (`T_out, Press_mm_hg, RH_out, Windspeed,
   Visibility, Tdewpoint`) are piecewise-linear between hourly nodes at HH:00
   (checked on the development region: second differences are non-zero only at
   rows at minute :10, i.e. kinks at HH:00). A value at HH:10 therefore encodes the
   HH+1:00 reading. **Only the value at the most recent HH:00 row `<= i` is used**
   (the real observation), never an interpolated row.
3. On-site sensors (`T1..T9, RH_1..RH_9, lights`) are 10-minute averages and are
   used at rows `<= i`. `T6`/`RH_6` (on-site outside sensor) is treated as weather
   but is a real 10-minute value.
4. Training examples are admitted only if the *target row* is `<=` the last
   labelled row of the training period (labels must exist before a model trained
   on them is used).
5. Scalers, medians and profiles are fitted on training rows only.
6. Row-timestamp convention (does row `t` cover `[t-10, t)` or `[t, t+10)`) is not
   documented. All analysis is consistent with either, because every feature and
   target is indexed by row and no feature uses a row after the origin.

## 4. Splits

| Region | Period | Use |
|---|---|---|
| Development | 2016-01-11 17:00 to 2016-04-22 23:50 | model/feature/hyper-parameter selection |
| Embargo | 2016-04-23 and 2016-04-24 | not used for training labels in the frozen model |
| **Final test** | **2016-04-25 00:00 to 2016-05-27 18:00** (32.75 days) | read once, after selection is frozen |

Development rolling-origin validation (4 expanding-window folds, Monday aligned):

| Fold | Validation origins | Training origins (target row must be before validation start) |
|---|---|---|
| V1 | 2016-02-29 to 2016-03-13 | from first usable row |
| V2 | 2016-03-14 to 2016-03-27 | from first usable row |
| V3 | 2016-03-28 to 2016-04-10 | from first usable row |
| V4 | 2016-04-11 to 2016-04-22 | from first usable row |

Final test access is controlled by a ledger (`results/final/test_access.json`):
the loader refuses to return test rows unless the ledger does not exist, and writes
it (with the git commit and the hash of this file) on first read. This is
discipline, not tamper-proofing; the git history is the audit trail.

## 5. Forecast inputs (feature sets)

Common to all sets, calendar features of the **target time**: sine/cosine
harmonics 1..3 of time of day (6) and day-of-week one-hot (7).

- **Set A** (own history): `y[i], y[i-1], y[i-2], y[i-3], y[i-5], y[i-11]`;
  means of `y` over the last 6, 36, 144 rows; std over the last 36;
  same-time-yesterday `y[i+h-144]` and mean of `y[i+h-150 .. i+h-144]`.
  (For h = 144 this is the last 7 rows; no later row is used.)
- **Set B** = A + `lights[i]`, mean of last 6 `lights`; `T1,T2,T3,T4,T5,T7,T8,T9`
  and `RH_1..RH_5, RH_7..RH_9` at `i`; indoor-mean temperature and its change over
  6 rows.
- **Set C** = B + airport-weather nodes (section 3 rule 2) for the six weather
  columns, 1-hour and 6-hour node changes in `T_out`, the mean `T_out` over the
  previous 24 nodes, `T6`, `RH_6`, and (indoor-mean minus `T6`).
- **Set N** (negative control only, never eligible) = C + `rv1`, `rv2`
  (the dataset's two random columns).

A weekly lag is deliberately excluded (it would discard the first week of data).

## 6. Models

Baselines (no fitting beyond a training-set statistic):
B1 persistence `y[i]`; B2 mean of last 6 rows; B3 same time yesterday
`y[i+h-144]`; B4 same time last week `y[i+h-1008]`; B5 median `y` for the target
slot of the day (training rows); B6 median `y` for the target slot of the week,
pooling the slot and its two neighbours.

Learned models (point forecasts, target is the median-oriented quantity because
MAE is the primary metric):

- Ridge on `log1p(y)`, standardised inputs, alpha in {1, 10, 100}.
- Random forest on `log1p(y)`: 300 trees, `max_features=0.5`,
  `min_samples_leaf` in {5, 20}.
- LightGBM, L1 objective on raw `y`: learning rate 0.05, `n_estimators` in
  {150, 400}, `num_leaves` in {7, 15}, `min_child_samples=40`,
  `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`,
  `reg_lambda=1`; the prediction is the mean of 3 seeds (0, 1, 2).

## 7. Selection (development folds only)

Metric: MAE averaged over the four folds. For each horizon:
the **headline model `M*`** is the (family, input set in {A,B,C}, hyper-parameters)
with the lowest mean fold MAE; the **study model `L*`** is the best LightGBM
configuration by the same rule (it carries the retraining and interval studies);
the **reference baseline `B*`** is the baseline with the lowest mean fold MAE.
No 1-SE rule, no manual override. The selection is written to
`results/dev/selection.json` and committed under tag `selection-frozen` before
the final test is read.

Dev sanity controls (reported, none can change the selection):
(a) set N must not beat set C by more than the fold-to-fold spread and `rv1/rv2`
must receive near-zero importance; (b) a model trained on **time-block-shuffled**
training targets must not beat B5 on validation; (c) seed-to-seed spread of the
LightGBM MAE.

## 8. Final-test experiments

**E1 Model vs baseline (headline).** `M*` and all baselines trained/fitted on the
development region only. Primary quantity per horizon: `MAE(M*) - MAE(B*)`.
Secondary: RMSE, and MAE for every baseline and every model/input-set pair
(full grid, labelled as not selected).

**E2 Input groups.** MAE of `L*`'s hyper-parameters under sets A, B, C on the
final test (descriptive; the selected set is whatever section 7 chose).

**E3 Retraining policies** (model `L*`; each retraining uses only rows whose
target is already observed at the retraining time; hyper-parameters fixed):
P0 frozen (dev only); P1 expanding, retrained every Monday 00:00; P2 sliding
28-day window, retrained every Monday 00:00; P3 drift-triggered: checked every day
at 00:00, retrain (expanding) when the shift statistic (section 9) is `>= 0.65`
and at least 3 days have passed since the last retrain; P4 expanding, retrained
every day 00:00 (cost upper reference). P3 threshold sensitivity {0.60, 0.70,
0.75} is reported as secondary only. Retrain counts are reported with every MAE.

**E4 Prediction intervals** (nominal 80 %, quantiles 0.1 / 0.9). `L*`
hyper-parameters, objective `quantile`. The quantile models are trained on
development rows excluding the last 14 days (and excluding rows whose targets
fall in them); those 14 days form the calibration set. Variants:
I0 raw quantiles; I1 static conformalised quantile regression (CQR) calibrated
once on the 14 calibration days; I2 rolling CQR, recalibrated every day 00:00
from the most recent 14 days of scores whose targets are observed; I3 adaptive
conformal (ACI, gamma = 0.01, target miscoverage 0.2) driven by the same scores
with the delayed-feedback rule (an outcome is usable only once its target row
`<= i`). Reported: coverage, mean width, interval score, coverage per test week.

**E5 Shift description (exploratory, no hypothesis test).** Per test week: mean
`T_out` change versus development, the shift statistic of section 9, the mean level
of `Appliances` versus development, and the errors of `B*`, `M*`, and the intervals.
Across the 33 test days, the Spearman correlation between daily shift statistic
and daily relative error is reported with a CI and labelled exploratory.

## 9. Shift statistic

Domain-classifier AUC: a small LightGBM (50 trees, 7 leaves) separates the
**inputs** of the "recent" window (origins in the last 7 days) from the
"reference" window (the model's most recent 28 days of training origins),
excluding calendar columns, with 3-fold cross-validation whose folds are whole
days. 0.5 means indistinguishable, 1.0 fully separable.

## 10. Statistics

- Errors are computed for every origin whose target row exists.
- Uncertainty: **day-block bootstrap** (resample calendar days of origins with
  replacement), 5,000 resamples, seed 20261015, percentile 95 % CI. All
  differences between two methods are paired (same resampled days).
  Sensitivity: moving-block bootstrap with blocks of 288 rows.
- Headline comparisons (E1, three horizons): also reported with the
  Bonferroni-adjusted 98.33 % CI.
- A difference is called supported only if its 95 % CI excludes zero. Otherwise
  the wording is "no evidence of a difference at this sample size".
- Coverage CIs use the same bootstrap. Nominal coverage is 0.80; "calibrated"
  means the CI contains 0.80.
- Test data are 33 days of one house: all intervals are about sampling noise in
  this house and period, not generalisation to other houses.

## 11. Directional expectations (not commitments)

H1: learned models beat the best baseline at h = 6 and h = 144 but not clearly at
h = 1 (persistence is strong at 10 min). H2: adding weather inputs does not help
much once own history is present. H3: periodic retraining helps modestly; daily
retraining adds little over weekly. H4: raw quantile intervals under-cover on the
shifted test period; rolling CQR / ACI move coverage toward 0.80. Any of these may
be wrong and will be reported either way.

## 12. Known limitations stated in advance

One house, one winter-to-spring window; the shift is gradual, not an abrupt
regime change; no public-holiday feature; no weather forecasts are available
(inputs are past observations only); correlated 10-minute errors make the
effective sample size far smaller than the number of rows.

## Deviations log

- 2026-10-15 (administrative, before any model run): the text above says the freeze is a git tag. The build sandbox cannot push tags (the git proxy drops tag pushes). The freeze marker is instead the commit `8d2bbb1` on branch `frozen/protocol` (pushed to GitHub). The same applies to `selection-frozen`, which becomes branch `frozen/selection`. No scientific content changed.
- 2026-10-15 **D2 (before any final-test read; found in a development-region dry run).**
  Section 9 defined the reference window as "the model's most recent 28 days of training
  origins" and the recent window as the last 7 days of origins. After a retrain, or right
  after the development period ends, these windows overlap, and removing the overlap leaves
  a recent window of only 1-3 days. On development data, a 3-fold day-grouped domain AUC with
  a 2-3 day positive window averaged 0.46 / 0.53 (k = 2 / 3 days) against 0.58 for a full
  7-day window, with values as low as 0.17 in the dry run: a known artefact of holding out
  most of a tiny positive class. **New definition:** reference = inputs of the 28 days
  *preceding* the recent window, i.e. origins in [T-35 d, T-7 d); recent = origins in
  [T-7 d, T). The statistic is now independent of any model and of retraining times.
  Everything else in section 9 (small LightGBM, 3-fold CV with whole-day folds, inputs of
  the selected model without calendar columns) and the P3 rule (>= 0.65, >= 3 days apart)
  is unchanged. No final-test value was read and no model selection had been made.
- 2026-10-15 **D3 (implementation notes, no change to any definition; before final-test read).**
  Synthetic-data smoke runs of `scripts/run_final.py` found three code bugs, all fixed with
  tests: a zero-length first model segment when the drift monitor fires on the first day, an
  empty prediction call on the last test day at h = 144 (no origin there has an observed
  target), and the shift statistic being recomputed per P3 threshold (now cached; it does not
  depend on the model). Implementation choices the protocol leaves open: a retraining at time
  T uses origins whose target row is strictly before T; lower interval bounds are clipped at
  0 Wh in every variant; ACI keeps alpha_t within [0.005, 0.5]; origins whose target row does
  not exist (the last h rows) are dropped everywhere.
