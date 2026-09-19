# Data contract

Raw data is intentionally excluded from version control.

The benchmark expects:

- `data/raw/sp500_headlines_2008_2024.csv`, with a date column and a headline/title column.
- `data/raw/fred_sp500.csv`, with `observation_date` and `SP500` columns.

The current case study uses:

1. **S&P 500 with Financial News Headlines (2008-2024)** by Dyuti Dasmahaptra, distributed on Kaggle under CC BY-SA 4.0. The raw file is not redistributed here.
2. **S&P 500 (SP500)** from FRED. FRED's available daily series begins in September 2016.

The code accepts common aliases such as `Title`, `Date`, `CP`, `headline`, `close`, and `observation_date`. The experiment joins headlines to the complete market calendar and never forward-fills text.

## Timing convention

Headline timestamps are only available at daily resolution. To avoid assuming that every day-t headline was observable before the day-t close, the default experiment applies a one-session execution lag:

- features are measured at the end of session `t`;
- the hypothetical position is entered at the close of `t+1`;
- the forecasted return runs from the close of `t+1` to the close of `t+2`.

This is conservative, explicit, and configurable.
