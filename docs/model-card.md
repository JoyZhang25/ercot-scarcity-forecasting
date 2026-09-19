# Model card

## Intended use

Research demonstration of leakage-resistant text-plus-market forecasting and temporal model evaluation.

## Out-of-scope uses

- Live trading or investment advice.
- Claims of causal news impact.
- Claims that a statistically weak test-set improvement is a persistent alpha.
- Predictions that assume intraday availability not supported by the source timestamps.

## Main limitations

- Daily headline dates do not identify the exact publication time.
- The headline dataset is a curated third-party sample and may contain selection bias, revisions, or duplicates not visible in the released fields.
- FRED's daily S&P 500 series begins in 2016, limiting the effective history.
- Hyperparameters are selected on one validation era; regime-specific robustness remains limited.
- The economic diagnostic omits bid-ask spreads, market impact, borrow constraints, taxes, and execution uncertainty.

## Reliability safeguards

- Conservative one-session execution lag.
- Outcome-date split boundaries.
- All learned transformations live inside scikit-learn pipelines.
- Locked test period.
- Hashes and package versions recorded in `run_manifest.json`.
- Paired block bootstrap and HAC inference for loss differences.
