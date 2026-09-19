# Information or Noise?

## Leakage-resistant S&P 500 forecasting with financial headlines

**[Project site](https://joyzhang25.github.io/market-news-signal/)** · **[Locked-test results](docs/results.md)** · **[Reconstruction audit](docs/reconstruction-audit.md)**

This project asks a narrow question: **do dated financial headlines add out-of-sample predictive information beyond simple market-history features?** It is an independent reconstruction of a 2025 Georgia Tech team course project, redesigned for chronological validity, reproducibility, and honest statistical interpretation.

The core contribution is the evaluation protocol, not a claim that a large neural network can beat the market.

## Current finding

On 542 held-out forecast dates from January 2022 through March 2024, the reconstruction finds **no reliable incremental predictive value from the released daily headlines**:

| Model | ROC-AUC | Balanced accuracy | Log loss |
|---|---:|---:|---:|
| Historical prior | 0.500 | 0.500 | 0.7022 |
| Market-only | 0.510 | 0.504 | 0.7160 |
| Text-only | 0.488 | 0.500 | 0.7023 |
| Market + text | 0.504 | 0.500 | 0.7160 |

The combined model's log-loss improvement over the market-only model is 0.00004, with a 95% moving-block-bootstrap interval of [-0.00111, 0.00112] and HAC p-value 0.94. Relative to the historical-prior forecast, the combined model is significantly worse on log loss. The evidence therefore does not support a forecasting or trading claim from this dataset and protocol.

That null result is the substantive result: once the target clock, learned transformations, model selection, and test set are controlled, the apparent signal does not survive. See [`docs/results.md`](docs/results.md) for the complete interpretation and [`docs/reconstruction-audit.md`](docs/reconstruction-audit.md) for a traceable account of what changed from the course pipeline.

## Research design

```text
headlines dated t + prices through close t
                    |
                    v
         market / text feature blocks
                    |
          one-session execution lag
                    |
                    v
       predict close(t+2) / close(t+1)
```

The one-session lag is deliberate: the headline source provides dates but not reliable publication times. Signals formed from all day-t headlines are therefore assumed tradable only at the close of t+1.

The benchmark compares four nested specifications on a locked chronological test set:

| Model | Information set |
|---|---|
| Prior | Historical up frequency |
| Market | Lagged return, 5-day momentum, 20-day volatility, news intensity |
| Text | Train-only TF-IDF headline features |
| Combined | Market and text blocks |

Regularization is selected on 2020-2021 data after training through 2019. Test evaluation begins in 2022. Preprocessing is encapsulated in scikit-learn pipelines so the vocabulary and scaling parameters cannot see validation or test observations during fitting.

## Evidence standard

The project reports:

- log loss and Brier score for probability quality;
- ROC-AUC, balanced accuracy, and macro-F1;
- calibration curves;
- moving-block-bootstrap confidence intervals for improvement over the market-only baseline;
- a Newey-West/HAC test of the paired loss differential;
- an explicitly illustrative, transaction-cost-adjusted long/short diagnostic.

A small metric improvement is described as **incremental predictive evidence** only when its uncertainty supports that reading. It is not presented as persistent alpha.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'

market-news-benchmark \
  --headlines data/raw/sp500_headlines_2008_2024.csv \
  --market data/raw/fred_sp500.csv \
  --config configs/benchmark.toml \
  --output outputs/benchmark

.venv/bin/pytest
```

Raw data is excluded from Git. See [`data/README.md`](data/README.md) for the schema, sources, licenses, and timing convention.

Data sources: [S&P 500 with Financial News Headlines (Kaggle)](https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024) and [FRED S&P 500](https://fred.stlouisfed.org/series/SP500).

## Repository structure

```text
configs/                 frozen experiment choices
docs/                    methodology and model card
src/market_news/         chronology, models, inference, and figures
tests/                   target-alignment and leakage tests
outputs/benchmark/       versioned locked-test metrics, predictions, figures, and manifest
```

## What changed from the course project

The original work explored sentence embeddings, HMM states, clustering, SMOTE, random forests, and LSTMs. That breadth was useful for learning, but several choices were not strong enough for a portfolio claim: random train/test splits appeared in parts of the pipeline, HMM smoothing obscured the forecast target, publication timing was not fully specified, and accuracy was reported without paired uncertainty or sufficiently strong baselines.

This repository is rewritten from scratch around one falsifiable question. It removes retrospective HMM labels, uses an observable future return, enforces chronological boundaries by outcome date, and makes the market-only ablation the central comparison.

## Disclaimer

For research and educational use only. Nothing in this repository is investment advice.
