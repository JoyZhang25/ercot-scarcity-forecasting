"""Leakage-resistant financial-news forecasting research package."""

from .config import ExperimentConfig
from .data import build_daily_panel, load_headlines, load_market, temporal_split

__all__ = [
    "ExperimentConfig",
    "build_daily_panel",
    "load_headlines",
    "load_market",
    "temporal_split",
]
