"""Publication-style project figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from .evaluation import strategy_returns


COLORS = {
    "prior": "#94a3b8",
    "market": "#0f766e",
    "text": "#c2410c",
    "combined": "#1d4ed8",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.dpi": 150,
        }
    )


def plot_model_comparison(metrics: pd.DataFrame, path: str | Path) -> None:
    _style()
    display = metrics.set_index("model").loc[["prior", "market", "text", "combined"]]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    specifications = [
        ("roc_auc", "ROC-AUC", (0.45, 0.70)),
        ("balanced_accuracy", "Balanced accuracy", (0.45, 0.70)),
        ("log_loss", "Log loss (lower is better)", None),
    ]
    for axis, (column, title, limits) in zip(axes, specifications):
        values = display[column]
        axis.bar(
            display.index,
            values,
            color=[COLORS[name] for name in display.index],
            width=0.72,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        if limits is not None:
            axis.set_ylim(*limits)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Held-out test performance", fontsize=14, fontweight="bold")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_diagnostics(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    transaction_cost_bps: float,
) -> None:
    _style()
    models = ["market", "text", "combined"]
    y_true = predictions["target_up"].to_numpy()
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
    for model in models:
        fpr, tpr, _ = roc_curve(y_true, predictions[f"p_{model}"])
        axes[0].plot(fpr, tpr, label=model, color=COLORS[model], linewidth=1.8)
    axes[0].set(title="ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0].legend(frameon=False)

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
    for model in models:
        observed, forecast = calibration_curve(
            y_true,
            predictions[f"p_{model}"],
            n_bins=8,
            strategy="quantile",
        )
        axes[1].plot(forecast, observed, marker="o", label=model, color=COLORS[model])
    axes[1].set(title="Probability calibration", xlabel="Forecast probability", ylabel="Observed frequency")

    for model in models:
        net = strategy_returns(
            predictions[f"p_{model}"].to_numpy(),
            predictions["target_return"].to_numpy(),
            transaction_cost_bps=transaction_cost_bps,
        )
        axes[2].plot(
            predictions["outcome_date"],
            np.exp(np.cumsum(net)),
            label=model,
            color=COLORS[model],
            linewidth=1.6,
        )
    axes[2].axhline(1.0, color="#94a3b8", linewidth=1)
    axes[2].set(title="Illustrative long/short wealth", xlabel="Outcome date", ylabel="Growth of $1")
    axes[2].tick_params(axis="x", rotation=25)
    figure.suptitle("Out-of-sample diagnostics", fontsize=14, fontweight="bold")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
