# Locked-test results

## Sample

- Training: 805 forecast dates, ending with outcomes on 2019-12-31.
- Validation: 503 forecast dates, ending with outcomes on 2021-12-31.
- Test: 542 forecast dates from 2022-01-03 through 2024-03-04.
- Source headlines after deduplication: 18,153.
- Complete S&P 500 market sessions used before feature/target trimming: 2,085.

The evaluation ends at the last observed headline date. Dates after the source stopped collecting headlines are not mislabeled as genuine zero-news days.

## Forecast performance

| Model | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Historical prior | 0.496 | 0.500 | 0.332 | 0.500 | 0.7022 | 0.2545 |
| Market-only | 0.500 | 0.504 | 0.343 | 0.510 | 0.7160 | 0.2610 |
| Text-only | 0.496 | 0.500 | 0.332 | 0.488 | 0.7023 | 0.2545 |
| Market + text | 0.496 | 0.500 | 0.338 | 0.504 | 0.7160 | 0.2610 |

The text-only model is effectively a regularized prior forecast on the locked test period. Its ranking performance is below chance. The combined model does not improve probability quality over either the prior or the market-only specification.

## Paired inference

The reported difference is baseline log loss minus candidate log loss, so a positive value favors the candidate.

| Baseline | Candidate | Mean difference | 95% block-bootstrap CI | HAC p-value |
|---|---|---:|---:|---:|
| Prior | Market | -0.01383 | [-0.02234, -0.00446] | 0.0012 |
| Prior | Text | -0.00008 | [-0.00124, 0.00114] | 0.8908 |
| Prior | Combined | -0.01379 | [-0.02215, -0.00469] | 0.0011 |
| Market | Combined | 0.00004 | [-0.00111, 0.00112] | 0.9404 |

The market and combined models are worse than the historical prior on log loss. Adding text to the market feature block produces an economically and statistically negligible paired change.

## Economic diagnostic

At five basis points per unit of turnover, the illustrative combined long/short rule has annualized log return 2.2%, annualized volatility 19.0%, Sharpe 0.12, and maximum drawdown -22.5%. The always-long prior has higher terminal wealth over the same period. These numbers are diagnostics, not a deployable backtest: the data lacks intraday timestamps and executable prices.

## Interpretation

The correct conclusion is not that financial language never matters. It is that this particular daily-resolution headline sample, conservative execution clock, feature set, and test period do not support the stronger claim made in the course report.

The result suggests three defensible extensions:

1. Acquire timestamped news and evaluate close-to-open or open-to-close horizons with an explicit publication cutoff.
2. Replace generic TF-IDF with pretrained financial-language features, while preserving the same locked test protocol.
3. Evaluate signal heterogeneity by volatility regime and news category using pre-registered subgroup definitions.

Any extension should retain the historical-prior and market-only baselines. A more complex model is useful only if it improves a proper scoring rule out of sample with uncertainty that excludes a negligible effect.
