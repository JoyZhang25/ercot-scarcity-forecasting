import numpy as np

from market_news.evaluation import (
    moving_block_indices,
    newey_west_mean_test,
    paired_block_bootstrap_log_loss,
)


def test_moving_block_indices_are_valid() -> None:
    indices = moving_block_indices(23, 5, np.random.default_rng(7))
    assert len(indices) == 23
    assert indices.min() >= 0
    assert indices.max() < 23


def test_bootstrap_detects_better_candidate() -> None:
    y = np.array([0, 1] * 60)
    baseline = np.full(len(y), 0.5)
    candidate = np.where(y == 1, 0.8, 0.2)
    result = paired_block_bootstrap_log_loss(
        y,
        baseline,
        candidate,
        reps=200,
        block_length=6,
        seed=3,
    )
    assert result["mean_log_loss_improvement"] > 0
    assert result["ci_2_5"] > 0


def test_newey_west_mean_test_reports_direction() -> None:
    result = newey_west_mean_test(np.linspace(0.01, 0.03, 80), max_lag=5)
    assert result["hac_mean"] > 0
    assert result["hac_t_stat"] > 0
