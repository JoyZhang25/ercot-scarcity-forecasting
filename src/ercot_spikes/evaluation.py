"""Rare-event, calibration, and economic-screening diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y: pd.Series, probability: np.ndarray, top_fraction: float
) -> dict[str, float]:
    """Evaluate discrimination and probability quality without accuracy theater."""
    alert_count = max(1, int(np.ceil(len(probability) * top_fraction)))
    alert = np.zeros(len(probability), dtype=bool)
    alert[np.argsort(probability, kind="stable")[-alert_count:]] = True
    return {
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "log_loss": log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6)),
        "precision_top_5pct": precision_score(y, alert, zero_division=0),
        "recall_top_5pct": recall_score(y, alert, zero_division=0),
        "base_rate": float(np.mean(y)),
    }


def risk_bins(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Summarize observed spikes and virtual-load spread by forecast risk decile."""
    ranked = frame.copy()
    ranked["risk_decile"] = pd.qcut(
        ranked["probability"].rank(method="first"), bins, labels=range(1, bins + 1)
    ).astype(int)
    return (
        ranked.groupby("risk_decile")
        .agg(
            observations=("spike", "size"),
            predicted_risk=("probability", "mean"),
            realized_spike_rate=("spike", "mean"),
            mean_rt_price=("rtm_price", "mean"),
            mean_rt_minus_da=("spread", "mean"),
        )
        .reset_index()
    )


def daily_block_interval(
    frame: pd.DataFrame, *, repetitions: int = 2000, seed: int = 7641
) -> tuple[float, float, float]:
    """Bootstrap top-decile minus unconditional RT-DA spread by operating day."""
    data = frame.copy()
    data["date"] = data.index.date
    data["top"] = data["probability"] >= data["probability"].quantile(0.9)
    data["top_spread"] = data["spread"].where(data["top"], 0.0)
    data["top_count"] = data["top"].astype(int)
    daily = data.groupby("date").agg(
        spread_sum=("spread", "sum"),
        count=("spread", "count"),
        top_spread_sum=("top_spread", "sum"),
        top_count=("top_count", "sum"),
    )
    estimate = float(
        daily.top_spread_sum.sum() / daily.top_count.sum()
        - daily.spread_sum.sum() / daily["count"].sum()
    )
    rng = np.random.default_rng(seed)
    selected_days = rng.integers(0, len(daily), size=(repetitions, len(daily)))
    arrays = daily.to_numpy()
    sampled = arrays[selected_days].sum(axis=1)
    boot = sampled[:, 2] / sampled[:, 3] - sampled[:, 0] / sampled[:, 1]
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)
