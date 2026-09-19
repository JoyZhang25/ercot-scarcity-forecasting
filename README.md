# Scarcity Before the Spike

### Forecasting ERCOT tail risk—and testing whether the forecast survives market prices

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-2F80ED.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-1B998B.svg)](.github/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-102A43.svg)](LICENSE)

**A market can be predictable without being mispriced.**

In ERCOT, electricity cannot wait on a shelf. A routine $30/MWh hour can become
a triple-digit scarcity event when weather-driven demand meets tight supply. This
project asks whether those hours can be identified *before* the day-ahead auction—and
then asks the harder question: did the auction already price the risk?

| Research question | Out-of-sample verdict | Key evidence |
|---|---|---|
| Can public pre-auction data rank next-day scarcity? | **Yes** | 2025 PR-AUC 0.095 vs 0.0268 base rate; 4.0× top-decile lift |
| Is the spike forecast itself a trading signal? | **No** | highest-risk decile earned a −$6.59/MWh mean RT−DA spread |
| Can a model trained directly on RT−DA find alpha? | **Candidate only** | positive 2026 lockbox P&L, but confidence and concentration tests fail |

That progression—**forecast risk, test monetization, reject what does not
survive**—is the point of the repository.

![Risk score separates scarcity from trading value](outputs/benchmark/figures/risk_lift.png)

## The information clock

Every prediction is made at **09:00 CT on operating day D−1**. The model may use
fixed 48-hour weather forecasts, calendar structure, and market history available
by then. It may not use target-day realized weather, load, or real-time price.
The scarcity classifier also excludes the target-day day-ahead settlement price.

| Period | Role | What it can influence |
|---|---|---|
| 2021–2023 | training | fitted parameters |
| 2024 | validation | model choice and probability calibration |
| 2025 | locked test | one evaluation of the scarcity hypothesis |
| 2026 YTD | prospective lockbox | one evaluation of the frozen alpha rule |

This clock matters more than model complexity. A powerful learner with revised
weather, contemporaneous load, or future prices would be an excellent backtest
and a useless forecast.

## 1 · Forecast the physical tail

The first target is deliberately operational:

> Will hourly HB_NORTH real-time price exceed **$100/MWh tomorrow**?

Only 235 of 8,759 hours crossed that threshold in 2025. An always-quiet model
would therefore be 97.3% accurate and practically worthless. Models are selected
by **precision–recall AUC**, then audited for ranking, calibration, and risk
concentration.

![Locked-test model comparison](outputs/benchmark/figures/model_comparison.png)

Gradient boosting won on 2024 validation data and was evaluated once on 2025:

- **0.095 PR-AUC**, 3.5× the random-ranking baseline;
- **0.816 ROC-AUC**;
- **10.62% realized spike rate** in the highest-risk decile, versus 2.68% overall;
- **0.0254 calibrated Brier score** after validation-only isotonic calibration.

The benchmark is intentionally broad but controlled: a seasonal prior,
regularized logistic regression, RBF SVM, random forest, histogram gradient
boosting, and a two-layer neural network all see the same chronological splits.
This is model comparison under a fixed research design, not a leaderboard search.

![Locked 2025 price and forecast-risk timeline](outputs/benchmark/figures/scarcity_timeline.png)

### What did the model learn?

The feature map combines three economic ideas: **forecast physical stress**
(48-hour GFS temperatures across five Texas cities), **market memory**
(strictly lagged prices, volatility, maxima, and spike frequency), and **seasonal
structure** (cyclical hour, weekday, and month).

![Held-out permutation importance](outputs/benchmark/figures/feature_importance.png)

Held-out permutation importance emphasizes time of day, seasonality, the same
hour one week earlier, the 48-hour price lag, volatility, and temperature
forecasts. A separate lagged-load ablation lowers validation PR-AUC from 0.084
to 0.079. More data did not mean more signal—and revision-prone annual load
archives were kept out of primary model selection.

## 2 · Ask whether the market already knew

A good scarcity forecast is not automatically alpha. The economic quantity is
the spread

```text
RT−DA = real-time settlement − day-ahead settlement.
```

The classifier's highest-risk decile did contain four times as many spikes, but
its mean RT−DA spread was **−$6.59/MWh**. Relative to all hours, its spread lift
was **−$4.14/MWh**, with a 95% daily block-bootstrap interval of
[−$7.67, −$0.42].

In plain English: the model recognized dangerous hours, but day-ahead prices
more than compensated for that danger. Reversing the trade after seeing the
result would be post-hoc storytelling, so the original monetization hypothesis
is recorded as a failure.

## 3 · Predict mispricing directly

The second experiment gives alpha a cleaner test. Instead of converting a spike
probability into a trade, a gradient-boosting regressor forecasts hourly RT−DA
directly. A virtual-supply position sells in the day-ahead market and buys back
in real time, so it profits when RT−DA is negative. The rule was committed in
[`0a7b554`](https://github.com/JoyZhang25/ercot-scarcity-forecasting/commit/0a7b554)
before 2026 outcomes were acquired:

> Each morning at 09:00 CT, select at most one next-day hour. Enter a 1 MW
> virtual-supply position only when predicted RT−DA ≤ −$3/MWh, then deduct a
> $2/MWh research hurdle.

| Walk-forward period | Role | Trades | Mean net P&L | Daily Sharpe | 95% daily-block CI |
|---|---|---:|---:|---:|---:|
| 2024 | validation | 177 | +$12.82/MWh | 1.20 | [−$9.16, +$32.15] |
| 2025 | shadow | 274 | +$4.83/MWh | 0.82 | [−$8.48, +$13.93] |
| **2026 YTD** | **prospective lockbox** | **111** | **+$13.76/MWh** | **0.88** | **[−$21.15, +$53.00]** |

![Prospective virtual-supply alpha audit](reports/figures/alpha_audit.png)

The 2026 point estimate is attractive: **$1,527 net** on 111 hypothetical 1 MW
positions, a **71.2% win rate**, and better performance than an equal-turnover
random date/hour baseline (randomization p=0.024). It is not yet credible alpha:

- the confidence interval crosses zero and the HAC t-statistic is 1.11;
- the five best days contribute 62.4% of positive P&L;
- excluding those days changes mean P&L to **−$7.01/MWh**;
- conditional on trading the same dates, hour selection beats random only at
  p=0.124.

**Verdict: positive but fragile candidate signal—not established alpha.** The
distinction is intentional. A backtest earns attention; robustness earns belief.

## Methods: machine learning first, market test second

This is not one flexible model carried from prediction into trading. The
scarcity study asks whether supervised learning can rank a physical tail event;
the alpha study asks whether any forecast survives prices, costs, and statistical
scrutiny.

| Layer | Methods actually used | Purpose |
|---|---|---|
| Supervised learning | class-weighted logistic regression, RBF SVM, random forest, histogram gradient boosting, two-layer MLP | compare linear, kernel, ensemble, and neural models under one design |
| Rare-event evaluation | PR-AUC, top-decile lift, precision/recall at fixed coverage, isotonic calibration, Brier score | avoid the false comfort of accuracy when spikes are only 2.68% of hours |
| Interpretation | held-out permutation importance and a lagged-load ablation | identify useful signal without reading importance off the training sample |
| Time-aware validation | point-in-time features, chronological selection, locked 2025 test, prospective 2026 lockbox | prevent look-ahead and repeated test-set tuning |
| Quant validation | direct RT−DA regression, explicit costs, equal-turnover randomization, HAC inference, daily block bootstrap, tail-concentration audit | distinguish predictive skill from economically robust alpha |

Together, those layers show the central result: **the same data can contain
forecasting signal without containing a defensible trading edge.**

The complete assumptions and rejection gates are documented in the
[research methodology](docs/methodology.md), [locked-test results](docs/results.md),
[model card](docs/model-card.md), and preregistered
[alpha protocol](docs/alpha_protocol.md). The
[research notebook](notebooks/ercot_price_spike_forecasting.ipynb) is the guided
walkthrough; reusable code lives in `src/ercot_spikes/` and is covered by tests.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'

# Download public ERCOT and weather inputs; raw files remain outside Git.
.venv/bin/python scripts/download_data.py

.venv/bin/ercot-spike-benchmark \
  --config configs/experiment.toml \
  --output outputs/benchmark

.venv/bin/ercot-alpha-research \
  --config configs/alpha_lockbox.toml \
  --output outputs/alpha

.venv/bin/pytest
```

See the [data contract](data/README.md) for source IDs, schemas, timing
assumptions, and the distinction between archived forecasts and realized system
variables.

<details>
<summary><strong>Repository map</strong></summary>

```text
configs/experiment.toml             frozen target, clock, split, and seed
configs/alpha_lockbox.toml           preregistered trading rule and alpha gate
notebooks/                           recruiter-readable research narrative
src/ercot_spikes/                    features, models, evaluation, and figures
tests/                               time-alignment and model-contract tests
outputs/benchmark/                   locked metrics, predictions, and figures
outputs/alpha/                       alpha metrics, baselines, and manifest
docs/                                methods, results, protocol, and limitations
```

</details>

## Scope

This is a reproducible research project, not investment advice or an ERCOT
operational tool. The virtual-supply study assumes a price-taking 1 MW position
that clears; it does not model QSE fees, collateral, uplift, bid-curve
non-clearance, or market impact. Results are not a promise of future performance.
