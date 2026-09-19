"""Frozen point-in-time research protocol for ERCOT virtual-supply alpha.

The module keeps prediction, position selection, and inference separate.  A
negative predicted ``RT - DA`` spread implies a virtual-supply position, whose
gross settlement is ``DA - RT``.  The implementation intentionally permits at
most one 1 MW position per operating day.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from .data import load_prices, load_weather


@dataclass(frozen=True)
class AlphaConfig:
    hub: str
    decision_hour_ct: int
    position_mw: float
    direction: str
    transaction_cost_per_mwh: float
    initial_train_end: str
    validation_year: int
    shadow_year: int
    lockbox_start: str
    lockbox_end: str
    predicted_spread_threshold: float
    max_positions_per_day: int
    target_clip: float
    random_state: int
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    hac_lags: int
    price_archive_dir: Path
    weather_dir: Path


def load_alpha_config(path: str | Path) -> AlphaConfig:
    """Load the immutable alpha-research specification."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    values = {
        **raw["research"],
        **raw["split"],
        **raw["signal"],
        **raw["model"],
        **raw["inference"],
        **{key: Path(value) for key, value in raw["paths"].items()},
    }
    config = AlphaConfig(**values)
    if config.direction != "virtual_supply":
        raise ValueError("The frozen protocol supports virtual_supply only")
    if config.max_positions_per_day != 1:
        raise ValueError("The frozen protocol requires one or fewer daily positions")
    if config.predicted_spread_threshold >= 0:
        raise ValueError("A virtual-supply threshold must be a negative RT-DA spread")
    return config


_WEATHER_FEATURES = [
    "temperature_2m_austin",
    "temperature_2m_dallas",
    "temperature_2m_houston",
    "temperature_2m_midland",
    "temperature_2m_sanantonio",
    "temp_mean",
    "temp_max",
    "temp_min",
    "cooling_degree",
    "heating_degree",
]


def build_alpha_frame(config: AlphaConfig) -> tuple[pd.DataFrame, list[str]]:
    """Construct the prespecified feature set without target-day leakage.

    All realized RT, DA, and spread histories end no later than operating day
    D-2.  The sole 24-hour lag is the prior operating day's *day-ahead* curve,
    which has already cleared before 09:00 CT on D-1.  Weather values are fixed
    48-hour forecast vintages rather than realized weather.
    """
    frame = load_prices(config.price_archive_dir, config.hub)
    frame = frame.join(load_weather(config.weather_dir), how="left")
    frame["spread"] = frame["rtm_price"] - frame["dam_price"]
    index = frame.index

    frame["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12)
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)

    features = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
    ]
    weather_features = [name for name in _WEATHER_FEATURES if name in frame]
    features.extend(weather_features)

    for lag in (48, 72, 168, 336, 720):
        for source, short_name in (
            ("rtm_price", "rt"),
            ("dam_price", "da"),
            ("spread", "spread"),
        ):
            name = f"{short_name}_lag_{lag}h"
            frame[name] = frame[source].shift(lag)
            features.append(name)

    frame["prior_day_da_same_hour"] = frame["dam_price"].shift(24)
    features.append("prior_day_da_same_hour")

    for source, short_name in (
        (frame["rtm_price"].shift(48), "rt"),
        (frame["dam_price"].shift(48), "da"),
        (frame["spread"].shift(48), "spread"),
    ):
        for days in (7, 30, 90):
            window = 24 * days
            minimum = max(48, window // 3)
            mean_name = f"{short_name}_mean_{days}d"
            volatility_name = f"{short_name}_vol_{days}d"
            frame[mean_name] = source.rolling(window, min_periods=minimum).mean()
            frame[volatility_name] = source.rolling(
                window, min_periods=minimum
            ).std()
            features.extend([mean_name, volatility_name])

    safe_spread = frame["spread"].shift(48)
    for days in (30, 90, 365):
        name = f"same_hour_spread_mean_{days}d"
        frame[name] = safe_spread.groupby(index.hour).transform(
            lambda values, window=days: values.rolling(
                window, min_periods=max(10, window // 3)
            ).mean()
        )
        features.append(name)

    frame = frame.loc[: pd.Timestamp(config.lockbox_end)]
    return frame, features


def alpha_model(config: AlphaConfig) -> Pipeline:
    """Return the prespecified robust, regularized nonlinear regressor."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=config.max_iter,
            learning_rate=config.learning_rate,
            max_leaf_nodes=config.max_leaf_nodes,
            l2_regularization=config.l2_regularization,
            random_state=config.random_state,
        ),
    )


def fit_and_predict(
    frame: pd.DataFrame,
    features: list[str],
    config: AlphaConfig,
    *,
    fit_end: str | pd.Timestamp,
    predict_start: str | pd.Timestamp,
    predict_end: str | pd.Timestamp,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit through ``fit_end`` and predict a strictly later interval."""
    fit_end = pd.Timestamp(fit_end)
    predict_start = pd.Timestamp(predict_start)
    predict_end = pd.Timestamp(predict_end)
    if fit_end >= predict_start:
        raise ValueError("fit_end must be strictly earlier than predict_start")
    train = frame.loc[:fit_end].dropna(subset=["spread"])
    prediction = frame.loc[predict_start:predict_end].dropna(subset=["spread"]).copy()
    model = alpha_model(config)
    target = train["spread"].clip(-config.target_clip, config.target_clip)
    model.fit(train[features], target)
    prediction["predicted_spread"] = model.predict(prediction[features])
    return model, prediction


def apply_frozen_rule(prediction: pd.DataFrame, config: AlphaConfig) -> pd.DataFrame:
    """Trade at most the single most-negative qualifying hour on each day."""
    result = prediction.copy()
    result["operating_date"] = result.index.date
    result["position_mw"] = 0.0
    eligible = result.loc[
        result["predicted_spread"] <= config.predicted_spread_threshold
    ]
    selected = (
        eligible.sort_values(["operating_date", "predicted_spread"])
        .groupby("operating_date", sort=False)
        .head(config.max_positions_per_day)
    )
    result.loc[selected.index, "position_mw"] = -config.position_mw
    result["gross_pnl"] = result["position_mw"] * result["spread"]
    result["net_pnl"] = result["gross_pnl"] - (
        result["position_mw"].abs() * config.transaction_cost_per_mwh
    )
    return result


def _daily_pnl(strategy: pd.DataFrame) -> pd.DataFrame:
    daily = strategy.groupby("operating_date").agg(
        net_pnl=("net_pnl", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        traded_mwh=("position_mw", lambda value: value.abs().sum()),
    )
    return daily


def hac_t_statistic(values: pd.Series, lags: int) -> float:
    """Newey-West t-statistic for the mean of a daily P&L series."""
    data = values.to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 3:
        return float("nan")
    centered = data - data.mean()
    long_run_variance = float(np.dot(centered, centered) / len(data))
    for lag in range(1, min(lags, len(data) - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / len(data))
        weight = 1.0 - lag / (lags + 1)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return float("nan")
    return float(data.mean() / np.sqrt(long_run_variance / len(data)))


def daily_block_bootstrap_interval(
    strategy: pd.DataFrame, config: AlphaConfig
) -> tuple[float, float]:
    """Bootstrap mean net P&L per traded MWh using operating-day blocks."""
    daily = _daily_pnl(strategy)
    rng = np.random.default_rng(config.bootstrap_seed)
    indices = rng.integers(
        0,
        len(daily),
        size=(config.bootstrap_repetitions, len(daily)),
    )
    values = daily[["net_pnl", "traded_mwh"]].to_numpy()
    sampled = values[indices].sum(axis=1)
    valid = sampled[:, 1] > 0
    distribution = sampled[valid, 0] / sampled[valid, 1]
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(low), float(high)


def strategy_metrics(strategy: pd.DataFrame, config: AlphaConfig) -> dict[str, float | int]:
    """Compute risk, stability, concentration, and inference diagnostics."""
    daily = _daily_pnl(strategy)
    traded = strategy.loc[strategy["position_mw"].ne(0)]
    cumulative = daily["net_pnl"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    monthly = daily.copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly_pnl = monthly["net_pnl"].resample("MS").sum()
    positive_days = daily.loc[daily["net_pnl"] > 0, "net_pnl"].sort_values(
        ascending=False
    )
    top_five_share = (
        float(positive_days.head(5).sum() / positive_days.sum())
        if positive_days.sum() > 0
        else float("nan")
    )
    traded_mwh = float(traded["position_mw"].abs().sum())
    interval_low, interval_high = daily_block_bootstrap_interval(strategy, config)
    daily_std = float(daily["net_pnl"].std())
    return {
        "observations": int(len(strategy)),
        "trades": int(len(traded)),
        "gross_pnl": float(traded["gross_pnl"].sum()),
        "net_pnl": float(traded["net_pnl"].sum()),
        "mean_net_pnl_per_mwh": float(traded["net_pnl"].sum() / traded_mwh),
        "mean_net_pnl_ci_95_low": interval_low,
        "mean_net_pnl_ci_95_high": interval_high,
        "win_rate": float((traded["net_pnl"] > 0).mean()),
        "annualized_daily_sharpe": float(
            daily["net_pnl"].mean() / daily_std * np.sqrt(365)
        ),
        "hac_t_statistic": hac_t_statistic(daily["net_pnl"], config.hac_lags),
        "max_drawdown": float(drawdown.min()),
        "positive_month_fraction": float((monthly_pnl > 0).mean()),
        "top_five_profit_days_share": top_five_share,
    }


def config_manifest(config: AlphaConfig) -> dict[str, object]:
    """Return a JSON-serializable copy of the frozen protocol."""
    result = asdict(config)
    result["price_archive_dir"] = str(config.price_archive_dir)
    result["weather_dir"] = str(config.weather_dir)
    return result
