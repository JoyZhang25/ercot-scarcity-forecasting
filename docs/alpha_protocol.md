# Prespecified alpha lockbox

Protocol frozen on **2026-09-19 before downloading or parsing 2026 settlement
outcomes**.

## Economic hypothesis

The scarcity classifier answers an operational question, but a buy-side alpha
must forecast a mispricing.  This extension therefore predicts the hourly
HB_NORTH convergence spread directly:

```text
spread = real-time settlement price - day-ahead settlement price
```

A negative conditional spread supports a **virtual-supply** position: sell 1 MW
in the day-ahead market and buy it back in real time.  Its gross settlement is
`DA - RT`.

The hypothesis is that the day-ahead auction sometimes overprices a small subset
of hours when forecast temperature stress, yesterday's cleared day-ahead curve,
and lagged market state jointly resemble historical overreaction regimes.  The
model is not allowed to observe the target day's day-ahead clearing price.

## Frozen signal

- decision time: 09:00 CT on operating day D-1;
- model: histogram gradient boosting with the parameters in
  `configs/alpha_lockbox.toml`;
- target: clipped hourly `RT - DA`, with clipping used only during fitting;
- eligible information: fixed 48-hour temperature-forecast vintages, calendar,
  the already-public D-1 day-ahead curve, and RT/DA/spread history ending no
  later than D-2;
- candidate trade: select the single most-negative forecast hour each day only
  when predicted `RT - DA <= -$3/MWh`;
- size: 1 MW virtual supply; no position on days without a qualifying hour;
- research friction reserve: $2/MWh per cleared position.

The $2/MWh deduction is a conservative hurdle, not a representation that every
participant faces a literal $2 exchange fee.  The backtest assumes a price-taking
offer that clears.  It does not model QSE fees, collateral, uplift allocation,
bid-curve non-clearance, market impact, or portfolio constraints, so even a
positive result is not a deployable P&L claim.

## Development evidence—not the verdict

The rule was selected using 2021-2023 for fitting and 2024 for validation.  A
year-boundary expanding refit produced a 2025 shadow result.  Both were positive
after the $2/MWh hurdle, but neither bootstrap interval excluded zero:

| Period | Trades | Mean net $/MWh | Daily Sharpe | 95% daily-block CI |
|---|---:|---:|---:|---:|
| 2024 validation | 177 | 12.82 | 1.20 | [-9.16, 32.15] |
| 2025 shadow | 274 | 4.83 | 0.82 | [-8.48, 13.93] |

The 2025 result is called *shadow*, not locked, because an earlier scarcity
diagnostic had already exposed the sign of high-risk spreads.  These numbers
justify a prospective test; they do not justify an alpha claim.

## Lockbox and rejection rule

The untouched lockbox is 2026-01-01 through 2026-09-12.  The model is refit once
through 2025, then the frozen daily rule is applied without retuning.

The repository will use the word **alpha** only if all of the following hold:

1. mean P&L remains positive after the $2/MWh hurdle;
2. the 95% operating-day block-bootstrap lower bound is above zero;
3. the HAC t-statistic and daily Sharpe support the same conclusion;
4. gains are not dominated by a handful of extreme days;
5. performance is not merely the unconditional ERCOT virtual-supply premium.

Failure of any gate is reported as a candidate signal or negative result.  No
threshold, direction, feature, or model may be changed after inspecting the 2026
outcome and still be described as the same lockbox test.
