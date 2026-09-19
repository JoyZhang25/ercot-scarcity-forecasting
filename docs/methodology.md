# Research protocol

## Estimand

For delivery hour \(t\), estimate

\[
p_t = \Pr(P_t^{RT} > 100 \mid \mathcal F_{09{:}00,\,D-1}),
\]

where \(P_t^{RT}\) is the hourly mean HB_NORTH real-time settlement price and
\(\mathcal F_{09{:}00,\,D-1}\) is the information set available at 09:00 CT on
the preceding operating day. This is a rare-event ranking and calibration problem,
not a next-price regression.

## Clock discipline

- GFS inputs are archived 48-hour-lead forecast vintages, never realized weather.
- RT price history enters at lags of 48 hours or more.
- Day-ahead price is excluded from the classifier because it is not known before
  the auction; it is used only after the fact to evaluate RT−DA spread.
- Every transformer is fit inside its training pipeline. Test data never determine
  scaling, imputation, model choice, or calibration.

The 48-hour guardrail is conservative. It absorbs publication delay and avoids the
common mistake of treating an operating-hour value as if it were instantly known.

## Features

The feature map has three coherent blocks:

1. **Forecast stress** — city-level temperature, statewide extrema, heating and
   cooling degrees from Dallas, Houston, Austin, San Antonio, and Midland.
2. **Market memory** — 48/72/168-hour RT lags; seven-day mean, volatility, and
   maximum; trailing 30-day spike frequency. All are shifted before rolling.
3. **Structure** — cyclical hour, weekday, and month encodings plus weekend status.

System and weather-zone native load form a named, 48-hour-lagged retrospective
ablation. Annual native-load archives can contain settlement revisions that were
not available at the original decision time, so this block is ineligible for
primary-model selection. It also lowers 2024 validation PR-AUC and is not carried
into the selected specification; the comparison remains in the versioned results.

## Model selection and locked test

| Period | Use |
|---|---|
| 2021–2023 | model fitting |
| 2024 | architecture selection and isotonic calibration |
| 2025 | one locked evaluation |

The controlled model family spans a smoothed seasonal prior, regularized logistic
regression, RBF SVM, random forest, histogram gradient boosting, and a two-layer
MLP. Selection uses 2024 precision–recall AUC. Hyperparameters are deliberately
compact; the repository demonstrates sound model comparison rather than a large
opaque search budget.

## Metrics

PR-AUC is primary because only 2.68% of 2025 hours cross the threshold. ROC-AUC,
log loss, Brier score, top-5%-risk precision, and recall provide complementary
views. Calibration is learned on 2024. An hourly accuracy number is intentionally
absent: an always-no-spike classifier would exceed 97% accuracy and be useless.

## Economic diagnostic

The pre-auction risk score is sorted into deciles and compared with the realized
virtual-load spread \(P^{RT}-P^{DA}\). The statistic is descriptive and receives a
daily block-bootstrap interval. It omits bid curves, market impact, uplift, credit,
fees, and execution constraints, so it is not labeled a tradable backtest.
