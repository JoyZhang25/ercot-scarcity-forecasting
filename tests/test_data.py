import numpy as np
import pandas as pd

from market_news.data import build_daily_panel, temporal_split


def _market(n: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 * np.exp(np.linspace(0.0, 0.2, n))
    return pd.DataFrame({"date": dates, "close": close})


def _headlines(market: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": market["date"], "headline": [f"headline-{i}" for i in range(len(market))]}
    )


def test_panel_stops_at_observed_headline_coverage() -> None:
    market = _market()
    headlines = _headlines(market.iloc[:60])
    panel = build_daily_panel(headlines, market, execution_lag_days=1)
    assert panel["outcome_date"].max() <= headlines["date"].max()


def test_target_respects_execution_lag() -> None:
    market = _market()
    panel = build_daily_panel(_headlines(market), market, execution_lag_days=1)
    row = panel.iloc[0]
    original_index = int(market.index[market["date"].eq(row["feature_date"])][0])
    expected = np.log(market.loc[original_index + 2, "close"] / market.loc[original_index + 1, "close"])
    assert row["execution_date"] == market.loc[original_index + 1, "date"]
    assert row["outcome_date"] == market.loc[original_index + 2, "date"]
    assert np.isclose(row["target_return"], expected)


def test_temporal_splits_do_not_cross_outcome_boundaries() -> None:
    market = _market(120)
    panel = build_daily_panel(_headlines(market), market, execution_lag_days=1)
    train_end = panel.iloc[35]["outcome_date"]
    validation_end = panel.iloc[60]["outcome_date"]
    split = temporal_split(panel, train_end=train_end, validation_end=validation_end)
    assert split.train["outcome_date"].max() <= train_end
    assert split.validation["feature_date"].min() > train_end
    assert split.validation["outcome_date"].max() <= validation_end
    assert split.test["feature_date"].min() > validation_end


def test_future_headline_change_does_not_change_past_features() -> None:
    market = _market()
    headlines = _headlines(market)
    original = build_daily_panel(headlines, market, execution_lag_days=1)
    changed = headlines.copy()
    changed.loc[changed.index[-1], "headline"] = "future-only-token"
    rebuilt = build_daily_panel(changed, market, execution_lag_days=1)
    cutoff = market.iloc[-5]["date"]
    columns = ["headline_text", "return_1d", "momentum_5d", "volatility_20d"]
    pd.testing.assert_frame_equal(
        original.loc[original.feature_date.lt(cutoff), columns].reset_index(drop=True),
        rebuilt.loc[rebuilt.feature_date.lt(cutoff), columns].reset_index(drop=True),
    )
