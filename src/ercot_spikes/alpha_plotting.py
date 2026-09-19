"""Publication-style figures for the prespecified alpha audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#17324D"
TEAL = "#0F766E"
RED = "#B54747"
AMBER = "#B7791F"
GRID = "#D9E2EC"
MUTED = "#627D98"
PAPER = "#F8FAFC"


def plot_alpha_audit(
    metrics: pd.DataFrame, predictions: pd.DataFrame, destination: str | Path
) -> None:
    """Show uncertainty, cumulative lockbox P&L, and monthly stability."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lockbox = predictions.loc[predictions["period"].eq("2026_lockbox")].copy()
    lockbox["operating_date"] = pd.to_datetime(lockbox["operating_date"])
    daily = lockbox.groupby("operating_date")["net_pnl"].sum()
    monthly = daily.resample("MS").sum()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    figure = plt.figure(figsize=(12.2, 7.3), facecolor=PAPER)
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.48, wspace=0.30)
    interval_axis = figure.add_subplot(grid[0, 0])
    cumulative_axis = figure.add_subplot(grid[0, 1])
    monthly_axis = figure.add_subplot(grid[1, :])
    for axis in (interval_axis, cumulative_axis, monthly_axis):
        axis.set_facecolor(PAPER)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID)

    labels = ["2024 validation", "2025 shadow", "2026 lockbox"]
    values = metrics["mean_net_pnl_per_mwh"].to_numpy()
    lows = metrics["mean_net_pnl_ci_95_low"].to_numpy()
    highs = metrics["mean_net_pnl_ci_95_high"].to_numpy()
    positions = np.arange(len(labels))[::-1]
    interval_axis.axvline(0, color=MUTED, linewidth=1)
    interval_axis.errorbar(
        values,
        positions,
        xerr=np.vstack([values - lows, highs - values]),
        fmt="o",
        color=TEAL,
        ecolor=NAVY,
        elinewidth=1.4,
        capsize=4,
        markersize=6,
    )
    interval_axis.set_yticks(positions, labels)
    interval_axis.set_xlabel("Mean net P&L ($/MWh), 95% day-block CI")
    interval_axis.set_title("Point estimates are positive; uncertainty still spans zero", loc="left", pad=10)
    interval_axis.grid(axis="x", color=GRID, linewidth=0.7)
    for value, position in zip(values, positions):
        interval_axis.annotate(
            f"${value:.2f}",
            (value, position),
            xytext=(6, 7),
            textcoords="offset points",
            color=NAVY,
            fontsize=9,
        )

    cumulative = daily.cumsum()
    cumulative_axis.plot(cumulative.index, cumulative.values, color=NAVY, linewidth=2)
    cumulative_axis.axhline(0, color=MUTED, linewidth=1)
    cumulative_axis.fill_between(
        cumulative.index, cumulative.values, 0, where=cumulative.values >= 0,
        color=TEAL, alpha=0.10,
    )
    cumulative_axis.set_title("2026 cumulative net P&L is tail-driven", loc="left", pad=10)
    cumulative_axis.set_ylabel("Net P&L ($), 1 MW")
    cumulative_axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    cumulative_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    cumulative_axis.grid(axis="y", color=GRID, linewidth=0.7)
    cumulative_axis.annotate(
        "Top five profit days = 62%\nof positive 2026 P&L",
        xy=(cumulative.index[-1], cumulative.iloc[-1]),
        xytext=(-118, -38),
        textcoords="offset points",
        color=RED,
        arrowprops={"arrowstyle": "-", "color": RED, "lw": 1},
        fontsize=9,
    )

    colors = [TEAL if value >= 0 else RED for value in monthly.values]
    bars = monthly_axis.bar(monthly.index, monthly.values, width=20, color=colors)
    monthly_axis.axhline(0, color=MUTED, linewidth=1)
    monthly_axis.set_title("Most 2026 months were positive, but January dominates", loc="left", pad=10)
    monthly_axis.set_ylabel("Monthly net P&L ($)")
    monthly_axis.xaxis.set_major_locator(mdates.MonthLocator())
    monthly_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    monthly_axis.grid(axis="y", color=GRID, linewidth=0.7)
    for bar, value in zip(bars, monthly.values):
        offset = 10 if value >= 0 else -14
        label = f"${value:,.0f}" if value >= 0 else f"-${abs(value):,.0f}"
        monthly_axis.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="center",
            color=NAVY,
            fontsize=8.5,
        )

    figure.suptitle(
        "Prospective virtual-supply audit: economically positive, statistically unproven",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )
    figure.text(
        0.07,
        0.02,
        "HB_NORTH · one 1 MW virtual-supply position/day at most · $2/MWh hurdle · rule frozen before 2026 outcomes",
        color=MUTED,
        fontsize=9,
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(figure)
