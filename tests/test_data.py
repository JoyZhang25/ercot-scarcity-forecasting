import pandas as pd
from pathlib import Path

from ercot_spikes.config import ExperimentConfig
from ercot_spikes.data import _hour_number, _timestamp, chronological_split


def config() -> ExperimentConfig:
    return ExperimentConfig(
        hub="HB_NORTH",
        spike_threshold=100.0,
        decision_hour_ct=9,
        minimum_lag_hours=48,
        train_end="2023-12-31 23:00",
        validation_end="2024-12-31 23:00",
        test_end="2025-12-31 23:00",
        random_state=7641,
        top_risk_fraction=0.05,
        price_archive_dir=Path("."),
        weather_dir=Path("."),
        load_archive_dir=Path("missing"),
    )


def test_hour_ending_maps_to_hour_beginning() -> None:
    dates = pd.Series(["01/02/2025", "01/02/2025"])
    hours = pd.Series([1, "24:00"])
    result = _timestamp(dates, hours)
    assert result.iloc[0] == pd.Timestamp("2025-01-02 00:00")
    assert result.iloc[1] == pd.Timestamp("2025-01-02 23:00")


def test_hour_parser_accepts_both_ercot_formats() -> None:
    assert _hour_number(pd.Series([1, "08:00", "24:00"])).tolist() == [1, 8, 24]


def test_chronological_split_has_no_overlap() -> None:
    index = pd.date_range("2023-12-31 22:00", "2025-01-01 01:00", freq="h")
    frame = pd.DataFrame({"x": range(len(index))}, index=index)
    train, validation, test = chronological_split(frame, config())
    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()
