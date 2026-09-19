from pathlib import Path

import numpy as np
import pandas as pd

from ercot_spikes.alpha import AlphaConfig, apply_frozen_rule, hac_t_statistic


def config() -> AlphaConfig:
    return AlphaConfig(
        hub="HB_NORTH",
        decision_hour_ct=9,
        position_mw=1.0,
        direction="virtual_supply",
        transaction_cost_per_mwh=2.0,
        initial_train_end="2023-12-31 23:00",
        validation_year=2024,
        shadow_year=2025,
        lockbox_start="2026-01-01 00:00",
        lockbox_end="2026-09-12 23:00",
        predicted_spread_threshold=-3.0,
        max_positions_per_day=1,
        target_clip=500.0,
        random_state=7641,
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=20.0,
        bootstrap_repetitions=100,
        bootstrap_seed=7641,
        hac_lags=7,
        price_archive_dir=Path("."),
        weather_dir=Path("."),
    )


def test_frozen_rule_selects_at_most_one_supply_hour_per_day() -> None:
    index = pd.date_range("2026-01-01", periods=48, freq="h")
    frame = pd.DataFrame(
        {
            "predicted_spread": np.r_[np.linspace(-10, 5, 24), np.linspace(-8, 6, 24)],
            "spread": 1.0,
        },
        index=index,
    )
    result = apply_frozen_rule(frame, config())
    traded = result.loc[result["position_mw"].ne(0)]
    assert (traded["position_mw"] == -1.0).all()
    assert traded.groupby("operating_date").size().max() == 1
    assert traded["net_pnl"].eq(-3.0).all()


def test_frozen_rule_respects_no_trade_threshold() -> None:
    index = pd.date_range("2026-01-01", periods=24, freq="h")
    frame = pd.DataFrame(
        {"predicted_spread": -2.99, "spread": -20.0}, index=index
    )
    result = apply_frozen_rule(frame, config())
    assert result["position_mw"].eq(0).all()
    assert result["net_pnl"].eq(0).all()


def test_hac_t_statistic_is_positive_for_positive_series() -> None:
    values = pd.Series([1.0, 0.5, 1.5, 0.75, 1.25] * 20)
    assert hac_t_statistic(values, lags=3) > 0
