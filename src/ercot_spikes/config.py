"""Experiment configuration with deliberately frozen research choices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    hub: str
    spike_threshold: float
    decision_hour_ct: int
    minimum_lag_hours: int
    train_end: str
    validation_end: str
    test_end: str
    random_state: int
    top_risk_fraction: float
    price_archive_dir: Path
    weather_dir: Path
    load_archive_dir: Path


def load_config(path: str | Path) -> ExperimentConfig:
    """Load the single experiment specification used by code and notebook."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    return ExperimentConfig(
        **raw["research"],
        **raw["split"],
        **raw["model"],
        **{key: Path(value) for key, value in raw["paths"].items()},
    )
