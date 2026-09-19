"""Readers and point-in-time feature construction for public ERCOT data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .config import ExperimentConfig


def _hour_number(values: pd.Series) -> pd.Series:
    """Parse ERCOT's numeric RT hour or ``HH:MM`` DAM hour-ending field."""
    text = values.astype(str).str.extract(r"(\d{1,2})", expand=False)
    return pd.to_numeric(text, errors="coerce")


def _timestamp(date: pd.Series, hour_ending: pd.Series) -> pd.Series:
    """Represent an ERCOT delivery interval by its local hour-beginning time."""
    return pd.to_datetime(date, format="%m/%d/%Y", errors="coerce") + pd.to_timedelta(
        _hour_number(hour_ending) - 1, unit="h"
    )


def read_price_archive(path: str | Path, market: str, hub: str) -> pd.DataFrame:
    """Read one annual ZIP and aggregate 15-minute RTM rows to hourly prices."""
    path = Path(path)
    market = market.lower()
    if market not in {"rtm", "dam"}:
        raise ValueError("market must be 'rtm' or 'dam'")
    with ZipFile(path) as archive:
        member = next(
            name for name in archive.namelist() if name.lower().endswith(".xlsx")
        )
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as workbook:
            workbook.write(archive.read(member))
            workbook.flush()
            excel = pd.ExcelFile(workbook.name, engine="calamine")
            chunks: list[pd.DataFrame] = []
            for sheet in excel.sheet_names:
                frame = pd.read_excel(workbook.name, sheet_name=sheet, engine="calamine")
                if market == "rtm":
                    frame = frame.loc[
                        frame["Settlement Point Name"].eq(hub),
                        ["Delivery Date", "Delivery Hour", "Settlement Point Price"],
                    ].rename(columns={"Delivery Hour": "hour"})
                else:
                    frame = frame.loc[
                        frame["Settlement Point"].eq(hub),
                        ["Delivery Date", "Hour Ending", "Settlement Point Price"],
                    ].rename(columns={"Hour Ending": "hour"})
                frame["timestamp"] = _timestamp(frame["Delivery Date"], frame["hour"])
                chunks.append(frame[["timestamp", "Settlement Point Price"]])
    result = pd.concat(chunks, ignore_index=True)
    column = f"{market}_price"
    result[column] = pd.to_numeric(result["Settlement Point Price"], errors="coerce")
    return (
        result.groupby("timestamp", as_index=False)[column]
        .mean()
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def load_prices(directory: str | Path, hub: str) -> pd.DataFrame:
    """Load all paired annual RTM and DAM price archives."""
    directory = Path(directory)
    parts: list[pd.DataFrame] = []
    for market in ("rtm", "dam"):
        annual = [
            read_price_archive(path, market, hub)
            for path in sorted(directory.glob(f"{market}_*.zip"))
        ]
        if not annual:
            raise FileNotFoundError(f"No {market}_YYYY.zip files found in {directory}")
        parts.append(pd.concat(annual).sort_index())
    return parts[0].join(parts[1], how="left")


def load_weather(directory: str | Path) -> pd.DataFrame:
    """Read fixed 48-hour-lead Open-Meteo GFS forecasts for Texas cities."""
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(directory).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            hourly = json.load(handle)["hourly"]
        city = path.stem.lower()
        frame = pd.DataFrame(hourly)
        frame["timestamp"] = pd.to_datetime(frame.pop("time"))
        frame = frame.set_index("timestamp")
        frame.columns = [
            column.replace("_previous_day2", "") + f"_{city}" for column in frame
        ]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No weather JSON files found in {directory}")
    weather = pd.concat(frames, axis=1).sort_index()
    groups = {
        "temp": [c for c in weather if c.startswith("temperature_2m_")],
        "humidity": [c for c in weather if c.startswith("relative_humidity_2m_")],
        "wind": [c for c in weather if c.startswith("wind_speed_10m_")],
        "solar": [c for c in weather if c.startswith("shortwave_radiation_")],
    }
    for name, columns in groups.items():
        if not columns:
            continue
        weather[f"{name}_mean"] = weather[columns].mean(axis=1)
        weather[f"{name}_max"] = weather[columns].max(axis=1)
        weather[f"{name}_min"] = weather[columns].min(axis=1)
    weather["cooling_degree"] = np.maximum(weather["temp_mean"] - 65.0, 0.0)
    weather["heating_degree"] = np.maximum(65.0 - weather["temp_mean"], 0.0)
    return weather


def load_actual_load(directory: str | Path) -> pd.DataFrame:
    """Read ERCOT's annual hourly native-load archives by weather zone."""
    annual: list[pd.DataFrame] = []
    for path in sorted(Path(directory).glob("load_*.zip")):
        with ZipFile(path) as archive:
            member = next(name for name in archive.namelist() if name.endswith(".xlsx"))
            with tempfile.NamedTemporaryFile(suffix=".xlsx") as workbook:
                workbook.write(archive.read(member))
                workbook.flush()
                frame = pd.read_excel(workbook.name, engine="calamine")
        timestamp = pd.to_datetime(frame.pop("Hour Ending"), errors="coerce") - pd.Timedelta(hours=1)
        frame.index = timestamp
        frame.index.name = "timestamp"
        frame = frame.rename(
            columns={
                "ERCOT": "actual_load",
                "COAST": "load_coast",
                "NCENT": "load_north_central",
                "FWEST": "load_far_west",
            }
        )
        annual.append(frame[["actual_load", "load_coast", "load_north_central", "load_far_west"]])
    if not annual:
        return pd.DataFrame()
    return pd.concat(annual).groupby(level=0).mean().sort_index()


def build_feature_frame(config: ExperimentConfig) -> tuple[pd.DataFrame, list[str]]:
    """Assemble labels, settlement outcomes, and strictly point-in-time features."""
    frame = load_prices(config.price_archive_dir, config.hub)
    frame = frame.join(load_weather(config.weather_dir), how="left")
    load = load_actual_load(config.load_archive_dir)
    if not load.empty:
        frame = frame.join(load, how="left")

    index = frame.index
    frame["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12)
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)

    lag = config.minimum_lag_hours
    for hours in (lag, 72, 168):
        frame[f"rt_price_lag_{hours}h"] = frame["rtm_price"].shift(hours)
    safe_history = frame["rtm_price"].shift(lag)
    frame["rt_mean_7d"] = safe_history.rolling(168, min_periods=72).mean()
    frame["rt_vol_7d"] = safe_history.rolling(168, min_periods=72).std()
    frame["rt_max_7d"] = safe_history.rolling(168, min_periods=72).max()
    frame["recent_spike_rate_30d"] = (
        safe_history.gt(config.spike_threshold).rolling(24 * 30, min_periods=168).mean()
    )

    for column in load.columns:
        frame[f"{column}_lag_{lag}h"] = frame[column].shift(lag)
    if not load.empty:
        safe_load = frame["actual_load"].shift(lag)
        frame["load_mean_7d"] = safe_load.rolling(168, min_periods=72).mean()
        frame["load_max_7d"] = safe_load.rolling(168, min_periods=72).max()
        frame["load_vol_7d"] = safe_load.rolling(168, min_periods=72).std()
    frame["spike"] = frame["rtm_price"].gt(config.spike_threshold).astype(int)

    excluded = {"rtm_price", "dam_price", "spike", *load.columns}
    features = [column for column in frame.columns if column not in excluded]
    frame = frame.loc[: pd.Timestamp(config.test_end)]
    return frame, features


def chronological_split(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return non-overlapping train, validation, and locked-test periods."""
    train_end = pd.Timestamp(config.train_end)
    validation_end = pd.Timestamp(config.validation_end)
    test_end = pd.Timestamp(config.test_end)
    train = frame.loc[:train_end]
    validation = frame.loc[train_end + pd.Timedelta(hours=1) : validation_end]
    test = frame.loc[validation_end + pd.Timedelta(hours=1) : test_end]
    return train, validation, test
