"""Data contracts and chronology-safe panel construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HEADLINE_ALIASES = ("headline", "title", "text")
DATE_ALIASES = ("date", "observation_date", "datetime")
CLOSE_ALIASES = ("close", "sp500", "cp", "adj close", "adj_close")

NUMERIC_FEATURES = (
    "return_1d",
    "momentum_5d",
    "volatility_20d",
    "news_intensity",
)


@dataclass(frozen=True)
class PanelSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...], kind: str) -> str:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    raise ValueError(f"Could not find a {kind} column. Available columns: {list(frame.columns)}")


def load_headlines(path: str | Path) -> pd.DataFrame:
    """Load dated headlines into the canonical ``date, headline`` schema."""

    raw = pd.read_csv(path)
    date_column = _find_column(raw, DATE_ALIASES, "date")
    headline_column = _find_column(raw, HEADLINE_ALIASES, "headline")
    frame = raw[[date_column, headline_column]].rename(
        columns={date_column: "date", headline_column: "headline"}
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["headline"] = frame["headline"].fillna("").astype(str).str.strip()
    frame = frame.dropna(subset=["date"])
    frame = frame.loc[frame["headline"].ne("")]
    return frame.drop_duplicates(["date", "headline"]).sort_values("date").reset_index(drop=True)


def load_market(path: str | Path) -> pd.DataFrame:
    """Load daily S&P 500 closes into the canonical ``date, close`` schema."""

    raw = pd.read_csv(path)
    date_column = _find_column(raw, DATE_ALIASES, "date")
    close_column = _find_column(raw, CLOSE_ALIASES, "close")
    frame = raw[[date_column, close_column]].rename(
        columns={date_column: "date", close_column: "close"}
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame.loc[frame["close"].gt(0)]
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def build_daily_panel(
    headlines: pd.DataFrame,
    market: pd.DataFrame,
    *,
    execution_lag_days: int = 1,
) -> pd.DataFrame:
    """Build one feature row per trading day without using future information.

    At feature date ``t`` all market features and dated headlines are known.
    With the default one-session execution lag, the position is entered at the
    close of trading session ``t+1`` and evaluated through the close of ``t+2``.
    This conservative clock avoids claiming that untimestamped day-t headlines
    were tradable at the day-t close.
    """

    if execution_lag_days < 0:
        raise ValueError("execution_lag_days must be non-negative")

    required_headline = {"date", "headline"}
    required_market = {"date", "close"}
    if not required_headline.issubset(headlines.columns):
        raise ValueError(f"Headlines must contain {sorted(required_headline)}")
    if not required_market.issubset(market.columns):
        raise ValueError(f"Market data must contain {sorted(required_market)}")

    daily_text = (
        headlines.assign(headline=headlines["headline"].fillna("").astype(str).str.strip())
        .loc[lambda x: x["headline"].ne("")]
        .groupby("date", as_index=False)
        .agg(
            headline_text=("headline", lambda values: " [SEP] ".join(values)),
            news_count=("headline", "size"),
        )
    )

    coverage_start = pd.Timestamp(headlines["date"].min())
    coverage_end = pd.Timestamp(headlines["date"].max())
    if pd.isna(coverage_start) or pd.isna(coverage_end):
        raise ValueError("Headline data has no valid coverage dates")

    # Outside this interval an empty merge could mean either "no news" or
    # "the source stopped collecting news."  Restricting the market calendar
    # to observed source coverage avoids silently treating missing coverage as
    # a genuine zero-news signal.
    panel = market.loc[
        market["date"].between(coverage_start, coverage_end), ["date", "close"]
    ].copy()
    panel = panel.sort_values("date").reset_index(drop=True)
    panel = panel.merge(daily_text, on="date", how="left", validate="one_to_one")
    panel["headline_text"] = panel["headline_text"].fillna("")
    panel["news_count"] = panel["news_count"].fillna(0).astype(int)

    log_close = np.log(panel["close"])
    panel["return_1d"] = log_close.diff(1)
    panel["momentum_5d"] = log_close.diff(5)
    panel["volatility_20d"] = panel["return_1d"].rolling(20, min_periods=20).std() * np.sqrt(252.0)
    panel["news_intensity"] = np.log1p(panel["news_count"].astype(float))

    entry_shift = execution_lag_days
    outcome_shift = execution_lag_days + 1
    panel["feature_date"] = panel["date"]
    panel["execution_date"] = panel["date"].shift(-entry_shift)
    panel["outcome_date"] = panel["date"].shift(-outcome_shift)
    panel["target_return"] = log_close.shift(-outcome_shift) - log_close.shift(-entry_shift)

    required = [*NUMERIC_FEATURES, "execution_date", "outcome_date", "target_return"]
    panel = panel.dropna(subset=required).copy()
    panel["target_up"] = panel["target_return"].gt(0).astype(int)

    ordered = [
        "feature_date",
        "execution_date",
        "outcome_date",
        "headline_text",
        "news_count",
        "close",
        *NUMERIC_FEATURES,
        "target_return",
        "target_up",
    ]
    return panel[ordered].sort_values("feature_date").reset_index(drop=True)


def temporal_split(
    panel: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> PanelSplit:
    """Create disjoint chronological splits using outcome dates at boundaries."""

    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    if train_end_ts >= validation_end_ts:
        raise ValueError("train_end must be before validation_end")

    train = panel.loc[panel["outcome_date"].le(train_end_ts)].copy()
    validation = panel.loc[
        panel["feature_date"].gt(train_end_ts)
        & panel["outcome_date"].le(validation_end_ts)
    ].copy()
    test = panel.loc[panel["feature_date"].gt(validation_end_ts)].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "Temporal split produced an empty partition: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )
    if train["outcome_date"].max() > train_end_ts:
        raise AssertionError("Training outcomes cross the training boundary")
    if validation["outcome_date"].max() > validation_end_ts:
        raise AssertionError("Validation outcomes cross the validation boundary")
    if test["feature_date"].min() <= validation_end_ts:
        raise AssertionError("Test features overlap the validation period")

    return PanelSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
