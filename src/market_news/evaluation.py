"""Forecast, statistical, and simple economic evaluation."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    prediction = (p >= 0.5).astype(int)
    return {
        "n_obs": float(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
    }


def strategy_returns(
    probability: np.ndarray,
    target_return: np.ndarray,
    *,
    transaction_cost_bps: float,
) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    returns = np.asarray(target_return, dtype=float)
    position = np.where(p >= 0.5, 1.0, -1.0)
    turnover = np.abs(np.diff(position, prepend=0.0))
    cost = turnover * transaction_cost_bps / 10_000.0
    return position * returns - cost


def strategy_metrics(
    probability: np.ndarray,
    target_return: np.ndarray,
    *,
    transaction_cost_bps: float,
) -> dict[str, float]:
    net = strategy_returns(
        probability,
        target_return,
        transaction_cost_bps=transaction_cost_bps,
    )
    position = np.where(np.asarray(probability) >= 0.5, 1.0, -1.0)
    turnover = np.abs(np.diff(position, prepend=0.0))
    volatility = float(np.std(net, ddof=1))
    sharpe = float(np.mean(net) / volatility * np.sqrt(252.0)) if volatility > 0 else 0.0
    wealth = np.exp(np.cumsum(net))
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "annualized_log_return": float(np.mean(net) * 252.0),
        "annualized_volatility": float(volatility * np.sqrt(252.0)),
        "sharpe": sharpe,
        "max_drawdown": float(np.min(drawdown)),
        "mean_daily_turnover": float(np.mean(turnover)),
        "terminal_wealth": float(wealth[-1]),
    }


def _pointwise_log_loss(y_true: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def moving_block_indices(
    n_obs: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    block = max(1, min(int(block_length), n_obs))
    blocks_needed = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs - block + 1, size=blocks_needed)
    return np.concatenate([np.arange(start, start + block) for start in starts])[:n_obs]


def paired_block_bootstrap_log_loss(
    y_true: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    *,
    reps: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    """CI for baseline log loss minus candidate log loss.

    A positive difference favors the candidate.
    """

    baseline_loss = _pointwise_log_loss(y_true, baseline_probability)
    candidate_loss = _pointwise_log_loss(y_true, candidate_probability)
    differential = baseline_loss - candidate_loss
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for rep in range(reps):
        idx = moving_block_indices(len(differential), block_length, rng)
        draws[rep] = float(np.mean(differential[idx]))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_log_loss_improvement": float(np.mean(differential)),
        "ci_2_5": float(lower),
        "ci_97_5": float(upper),
        "probability_improvement_positive": float(np.mean(draws > 0)),
    }


def newey_west_mean_test(values: np.ndarray, *, max_lag: int) -> dict[str, float]:
    """Two-sided HAC test that a loss differential has zero mean."""

    x = np.asarray(values, dtype=float)
    n_obs = len(x)
    if n_obs < 3:
        raise ValueError("At least three observations are required")
    centered = x - np.mean(x)
    lag_cap = min(max(int(max_lag), 0), n_obs - 1)
    long_run_variance = float(np.dot(centered, centered) / n_obs)
    for lag in range(1, lag_cap + 1):
        weight = 1.0 - lag / (lag_cap + 1.0)
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / n_obs)
        long_run_variance += 2.0 * weight * autocovariance
    standard_error = sqrt(max(long_run_variance, 0.0) / n_obs)
    statistic = float(np.mean(x) / standard_error) if standard_error > 0 else 0.0
    p_value = float(erfc(abs(statistic) / sqrt(2.0)))
    return {
        "hac_mean": float(np.mean(x)),
        "hac_t_stat": statistic,
        "hac_p_value": p_value,
        "hac_max_lag": float(lag_cap),
    }


def log_loss_differential(
    y_true: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> np.ndarray:
    return _pointwise_log_loss(y_true, baseline_probability) - _pointwise_log_loss(
        y_true, candidate_probability
    )
