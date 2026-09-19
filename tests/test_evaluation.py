import numpy as np
import pandas as pd

from ercot_spikes.evaluation import classification_metrics, risk_bins


def test_top_risk_metrics_reward_correct_ranking() -> None:
    y = pd.Series([0, 0, 0, 1, 1])
    probability = np.array([0.01, 0.02, 0.03, 0.8, 0.9])
    metrics = classification_metrics(y, probability, top_fraction=0.4)
    assert metrics["pr_auc"] == 1.0
    assert metrics["recall_top_5pct"] == 1.0


def test_risk_bins_preserve_observations() -> None:
    index = pd.date_range("2025-01-01", periods=20, freq="h")
    frame = pd.DataFrame(
        {
            "probability": np.linspace(0.01, 0.9, 20),
            "spike": [0] * 18 + [1, 1],
            "rtm_price": np.arange(20),
            "spread": np.arange(20) - 5,
        },
        index=index,
    )
    result = risk_bins(frame, bins=5)
    assert result["observations"].sum() == 20
    assert result.iloc[-1]["realized_spike_rate"] > result.iloc[0]["realized_spike_rate"]
