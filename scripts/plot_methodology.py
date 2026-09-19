"""Render the README's point-in-time research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


NAVY = "#17324D"
BLUE = "#2F80ED"
TEAL = "#0F766E"
AMBER = "#B7791F"
RED = "#B54747"
MUTED = "#627D98"
GRID = "#D9E2EC"
PAPER = "#FFFFFF"
PALE = "#F5F8FB"


def _step(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    number: int,
    title: str,
    lines: tuple[str, ...],
    color: str,
) -> None:
    """Draw one editorial-style step without a heavy card border."""
    axis.add_patch(Circle((x, y), 0.018, facecolor=color, edgecolor="none"))
    axis.text(
        x,
        y,
        str(number),
        ha="center",
        va="center",
        color=PAPER,
        fontsize=9,
        fontweight="bold",
    )
    text_x = x + 0.030
    axis.text(
        text_x,
        y + 0.016,
        title,
        ha="left",
        va="bottom",
        color=NAVY,
        fontsize=10.3,
        fontweight="bold",
    )
    axis.text(
        text_x,
        y - 0.006,
        "\n".join(lines),
        ha="left",
        va="top",
        color=MUTED,
        fontsize=8.8,
        linespacing=1.35,
    )


def plot_methodology(destination: str | Path) -> None:
    """Show how point-in-time data become two distinct empirical tests."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "figure.facecolor": PAPER,
        }
    )

    figure, axis = plt.subplots(figsize=(13.6, 7.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.055,
        0.945,
        "One information set, two different research questions",
        color=NAVY,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axis.text(
        0.055,
        0.905,
        "Forecast physical scarcity first; claim alpha only if the forecast survives market prices and a prospective audit.",
        color=MUTED,
        fontsize=10.5,
        ha="left",
        va="top",
    )

    shared = FancyBboxPatch(
        (0.055, 0.780),
        0.890,
        0.090,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=PALE,
        edgecolor=GRID,
        linewidth=1.0,
    )
    axis.add_patch(shared)
    axis.text(
        0.075,
        0.842,
        "DECISION CLOCK",
        color=AMBER,
        fontsize=8.5,
        fontweight="bold",
        va="center",
    )
    axis.text(
        0.075,
        0.810,
        "09:00 CT on D−1",
        color=NAVY,
        fontsize=10.3,
        fontweight="bold",
        va="center",
    )
    axis.plot([0.225, 0.225], [0.795, 0.855], color=GRID, linewidth=1.0)
    axis.text(
        0.245,
        0.842,
        "POINT-IN-TIME INPUTS",
        color=AMBER,
        fontsize=8.5,
        fontweight="bold",
        va="center",
    )
    axis.text(
        0.245,
        0.810,
        "48h GFS vintages · calendar · realized market history ending no later than D−2",
        color=NAVY,
        fontsize=9.7,
        va="center",
    )
    axis.plot([0.725, 0.725], [0.795, 0.855], color=GRID, linewidth=1.0)
    axis.text(
        0.745,
        0.842,
        "CHRONOLOGY",
        color=AMBER,
        fontsize=8.5,
        fontweight="bold",
        va="center",
    )
    axis.text(
        0.745,
        0.810,
        "Fit 2021–23 · select 2024 · lock 2025/26",
        color=NAVY,
        fontsize=9.7,
        va="center",
    )

    axis.plot([0.500, 0.500], [0.110, 0.735], color=GRID, linewidth=1.0)
    axis.plot([0.500, 0.500], [0.780, 0.750], color=GRID, linewidth=1.2)
    axis.plot([0.265, 0.735], [0.750, 0.750], color=GRID, linewidth=1.2)
    axis.plot([0.265, 0.265], [0.750, 0.728], color=GRID, linewidth=1.2)
    axis.plot([0.735, 0.735], [0.750, 0.728], color=GRID, linewidth=1.2)

    axis.text(
        0.055,
        0.720,
        "A  ·  FORECAST PHYSICAL SCARCITY",
        color=BLUE,
        fontsize=11.5,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.055,
        0.687,
        "Classification target  yₕ = 1{RTₕ > $100/MWh}",
        color=NAVY,
        fontsize=9.5,
        va="top",
    )
    axis.text(
        0.535,
        0.720,
        "B  ·  FORECAST MARKET MISPRICING",
        color=TEAL,
        fontsize=11.5,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.535,
        0.687,
        "Regression target  sₕ = RTₕ − DAₕ",
        color=NAVY,
        fontsize=9.5,
        va="top",
    )

    left_x = 0.077
    right_x = 0.557
    step_y = (0.610, 0.475, 0.340, 0.205)

    _step(
        axis,
        x=left_x,
        y=step_y[0],
        number=1,
        title="Build a leakage-safe hourly panel",
        lines=(
            "Weather stress + seasonal structure + RT lags/rolling statistics.",
            "All realized price and load variables are shifted at least 48 hours.",
        ),
        color=BLUE,
    )
    _step(
        axis,
        x=left_x,
        y=step_y[1],
        number=2,
        title="Run a controlled model tournament",
        lines=(
            "Prior · logit · RBF SVM · random forest · gradient boosting · MLP.",
            "One feature set and split; select only by 2024 validation PR-AUC.",
        ),
        color=BLUE,
    )
    _step(
        axis,
        x=left_x,
        y=step_y[2],
        number=3,
        title="Calibrate, freeze, then open 2025 once",
        lines=(
            "Validation-only isotonic map; audit PR-AUC, Brier score, lift,",
            "and held-out permutation importance on 235 spikes / 8,759 hours.",
        ),
        color=BLUE,
    )
    _step(
        axis,
        x=left_x,
        y=step_y[3],
        number=4,
        title="Translate risk ranking into a pricing test",
        lines=(
            "Sort calibrated risk into deciles and compare realized RT−DA.",
            "Top decile: 10.62% spikes, but −$6.59/MWh mean spread → no alpha.",
        ),
        color=BLUE,
    )

    _step(
        axis,
        x=right_x,
        y=step_y[0],
        number=1,
        title="Model the economic target directly",
        lines=(
            "Regularized histogram gradient boosting estimates clipped E[RT−DA | F].",
            "Adds lagged DA/spread history and the prior day's cleared DA curve.",
        ),
        color=TEAL,
    )
    _step(
        axis,
        x=right_x,
        y=step_y[1],
        number=2,
        title="Freeze a sparse virtual-supply rule",
        lines=(
            "Choose at most one next-day hour if predicted RT−DA ≤ −$3/MWh.",
            "Sell 1 MW DA, buy it back RT, and deduct a $2/MWh hurdle.",
        ),
        color=TEAL,
    )
    _step(
        axis,
        x=right_x,
        y=step_y[2],
        number=3,
        title="Walk forward without revisiting the rule",
        lines=(
            "2024 validation → 2025 shadow → 2026 prospective lockbox.",
            "Refit at each year boundary using only data available beforehand.",
        ),
        color=TEAL,
    )
    _step(
        axis,
        x=right_x,
        y=step_y[3],
        number=4,
        title="Require the signal to survive robustness gates",
        lines=(
            "Daily-block CI crosses zero; HAC t = 1.11; top five days = 62.4%.",
            "Mean excluding those days = −$7.01/MWh → candidate, not alpha.",
        ),
        color=TEAL,
    )

    verdict = FancyBboxPatch(
        (0.055, 0.055),
        0.890,
        0.070,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor="#FFF8ED",
        edgecolor="#E8D4AE",
        linewidth=1.0,
    )
    axis.add_patch(verdict)
    axis.text(
        0.075,
        0.090,
        "RESULT",
        color=AMBER,
        fontsize=8.5,
        fontweight="bold",
        va="center",
    )
    axis.text(
        0.145,
        0.090,
        "Scarcity is forecastable; the available evidence does not yet establish a durable trading edge.",
        color=NAVY,
        fontsize=10.5,
        fontweight="bold",
        va="center",
    )

    figure.savefig(destination, bbox_inches="tight", facecolor=PAPER)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="reports/figures/methodology_pipeline.svg",
        help="Destination SVG or PNG path.",
    )
    args = parser.parse_args()
    plot_methodology(args.output)


if __name__ == "__main__":
    main()
