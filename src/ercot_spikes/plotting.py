"""Portfolio-quality figures with one consistent visual grammar."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

NAVY = "#102A43"
BLUE = "#2F80ED"
ORANGE = "#F2994A"
RED = "#D64545"
TEAL = "#1B998B"
GREY = "#829AB1"
PALE = "#EAF2F8"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelcolor": NAVY,
            "axes.edgecolor": "#BCCCDC",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#486581",
            "ytick.color": "#486581",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_timeline(predictions: pd.DataFrame, path: Path) -> None:
    """Show the rare-event target and the model's ex-ante risk score together."""
    _style()
    daily = predictions.resample("D").agg(
        rt_price=("rtm_price", "max"), probability=("probability", "max")
    )
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(daily.index, daily.rt_price, color=PALE, alpha=1)
    ax.plot(daily.index, daily.rt_price, color=NAVY, lw=1.1, label="Daily maximum RT price")
    ax.axhline(100, color=RED, ls="--", lw=1, label="$100/MWh spike threshold")
    ax.set_ylim(0, min(600, max(250, daily.rt_price.quantile(0.99) * 1.15)))
    ax.set_ylabel("HB_NORTH real-time price ($/MWh)")
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily.probability, color=ORANGE, lw=1.15, alpha=0.9, label="Max predicted risk")
    ax2.set_ylabel("Predicted spike probability", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.set_ylim(0, max(0.12, daily.probability.quantile(0.995) * 1.1))
    ax.set_title("Can tomorrow's dangerous hours be identified before the day-ahead auction?", loc="left", pad=34)
    ax.text(0, 1.015, "Locked 2025 test year · forecasts and outcomes shown at daily maxima", transform=ax.transAxes, color=GREY)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False, ncol=3)
    _save(fig, path)


def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    """Compare rare-event ranking skill; the dashed line is the no-skill rate."""
    _style()
    ordered = metrics.sort_values("pr_auc")
    colors = [
        GREY if name == "Seasonal prior" else ORANGE if name == "Gradient boosting" else BLUE
        for name in ordered.model
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(ordered.model, ordered.pr_auc, color=colors, height=0.62)
    base = float(ordered.base_rate.iloc[0])
    ax.axvline(base, color=RED, ls="--", lw=1.2, label=f"Random ranking = {base:.1%}")
    for bar, value in zip(bars, ordered.pr_auc):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", color=NAVY)
    ax.set_xlim(0, max(ordered.pr_auc.max() * 1.2, base * 2.5))
    ax.set_xlabel("Precision–recall AUC")
    ax.set_title("Nonlinear models are judged on rare-event ranking, not accuracy", loc="left", pad=34)
    ax.text(0, 1.015, "All models evaluated once on the same locked 2025 test year", transform=ax.transAxes, color=GREY)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def plot_risk_lift(bins: pd.DataFrame, path: Path) -> None:
    """Connect statistical ranking to an interpretable market outcome."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(bins.risk_decile, bins.realized_spike_rate * 100, color=[GREY] * 9 + [ORANGE], alpha=0.95)
    ax.set_xlabel("Predicted risk decile (10 = highest)")
    ax.set_ylabel("Realized spike rate (%)")
    ax.set_xticks(range(1, 11))
    ax2 = ax.twinx()
    ax2.plot(bins.risk_decile, bins.mean_rt_minus_da, color=TEAL, marker="o", lw=2)
    ax2.axhline(0, color="#9FB3C8", lw=0.8)
    ax2.set_ylabel("Mean RT − DA spread ($/MWh)", color=TEAL)
    ax2.tick_params(axis="y", colors=TEAL)
    ax.set_title("The risk score concentrates spikes—but not automatic trading alpha", loc="left", pad=34)
    ax.text(0, 1.015, "RT−DA spread is evaluated separately from operational tail risk", transform=ax.transAxes, color=GREY)
    _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    """Plot held-out permutation importance for the selected model."""
    _style()
    top = importance.nlargest(12, "importance_mean").sort_values("importance_mean")
    labels = top.feature.str.replace("_", " ").str.replace("temperature 2m", "temperature")
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.barh(labels, top.importance_mean, xerr=top.importance_std, color=BLUE, alpha=0.9, ecolor="#9FB3C8")
    ax.axvline(0, color=NAVY, lw=0.8)
    ax.set_xlabel("Decrease in test PR-AUC after permutation")
    ax.set_title("What the model uses: weather stress, seasonality, and market memory", loc="left", pad=34)
    ax.text(0, 1.015, "Permutation importance on the untouched 2025 test year", transform=ax.transAxes, color=GREY)
    _save(fig, path)


def plot_calibration(y: pd.Series, probability: np.ndarray, path: Path) -> None:
    """Reliability diagram for the calibrated selected model."""
    _style()
    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", label="Perfect calibration")
    ax.plot(predicted, observed, color=BLUE, marker="o", lw=2, label="Selected model")
    limit = max(0.2, observed.max() * 1.15, predicted.max() * 1.15)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed spike frequency")
    ax.set_title("A risk score should mean what it says")
    ax.legend(frameon=False)
    _save(fig, path)
