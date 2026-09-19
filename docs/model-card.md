# Model card

## Intended use

Research prototype for ranking next-day ERCOT HB_NORTH hours by the probability of
an hourly real-time price above $100/MWh before the day-ahead auction. Appropriate
uses include interview discussion, model-risk review, and reproducible research.

## Not intended for

Live dispatch, autonomous bidding, reliability operations, or investment advice.
The model does not represent nodal congestion, bid-stack microstructure, outages,
market impact, or participant-specific costs.

## Selected specification

Histogram gradient boosting was selected on 2024 validation PR-AUC, then calibrated
with validation-year isotonic regression. The untouched 2025 test contains 8,759
hours and 235 labeled spikes. A retrospective lagged native-load feature block was
tested, marked selection-ineligible because of possible settlement revisions, and
rejected on validation rather than silently retained.

## Material limitations

- A hub price suppresses nodal congestion structure.
- Weather forecasts are sampled at five cities, not every ERCOT weather zone.
- Extreme-weather and policy regimes can shift faster than the training window.
- The $100 threshold is economically interpretable but not a regulatory definition
  of scarcity.
- Economic value is not implied by classification skill. The RT−DA diagnostic
  rejects the prespecified virtual-load direction in 2025, but it is not a complete
  executable P&L study and does not justify a post-hoc reversed trade.

## Monitoring

A production version should track calibration drift, PR-AUC by season, feature
coverage, forecast-vintage latency, threshold sensitivity, and performance during
named stress events. Retraining must preserve an expanding or rolling time split.
