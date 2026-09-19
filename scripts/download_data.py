#!/usr/bin/env python3
"""Download public ERCOT prices, native load, and point-in-time weather."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

PRICE_DOCUMENTS = {
    "rtm": {
        2021: 814922832,
        2022: 886632075,
        2023: 969805139,
        2024: 1065471230,
        2025: 1177737535,
        # Snapshot posted 2026-09-13; frozen lockbox ends 2026-09-12.
        2026: 1274016263,
    },
    "dam": {
        2021: 814918746,
        2022: 886627599,
        2023: 969803138,
        2024: 1065468714,
        2025: 1177667469,
        # Snapshot posted 2026-09-13; frozen lockbox ends 2026-09-12.
        2026: 1274009269,
    },
}

CITIES = {
    "Dallas": (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Austin": (30.2672, -97.7431),
    "SanAntonio": (29.4241, -98.4936),
    "Midland": (31.9973, -102.0779),
}

LOAD_ARCHIVES = {
    2021: "https://www.ercot.com/files/docs/2021/11/12/Native_Load_2021.zip",
    2022: "https://www.ercot.com/files/docs/2022/02/08/Native_Load_2022.zip",
    2023: "https://www.ercot.com/files/docs/2023/02/09/Native_Load_2023.zip",
    2024: "https://www.ercot.com/files/docs/2024/02/06/Native_Load_2024.zip",
    2025: "https://www.ercot.com/files/docs/2025/02/11/Native_Load_2025.zip",
}


def _get(url: str, *, params: dict | list[tuple[str, str]] | None = None) -> requests.Response:
    for attempt in range(7):
        response = requests.get(
            url,
            params=params,
            timeout=180,
            headers={"User-Agent": "ercot-scarcity-forecasting/0.2 research"},
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait = int(response.headers.get("Retry-After", 0)) or min(60, 5 * 2**attempt)
        time.sleep(wait)
    response.raise_for_status()
    raise AssertionError("unreachable")


def download_prices(root: Path) -> None:
    destination = root / "ercot" / "archives"
    destination.mkdir(parents=True, exist_ok=True)
    base = "https://www.ercot.com/misdownload/servlets/mirDownload"
    for market, years in PRICE_DOCUMENTS.items():
        for year, document in years.items():
            path = destination / f"{market}_{year}.zip"
            if not path.exists():
                path.write_bytes(_get(base, params={"doclookupId": document}).content)
            print(f"price: {path}")


def download_weather(root: Path, *, end_date: str, refresh: bool = False) -> None:
    destination = root / "ercot" / "weather"
    destination.mkdir(parents=True, exist_ok=True)
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    for city, (latitude, longitude) in CITIES.items():
        path = destination / f"{city}.json"
        if refresh or not path.exists():
            parameters = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": "2021-01-01",
                "end_date": end_date,
                "hourly": ",".join(
                    [
                        "temperature_2m_previous_day2",
                    ]
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "America/Chicago",
                "models": "gfs_seamless",
            }
            path.write_text(json.dumps(_get(url, params=parameters).json()), encoding="utf-8")
            time.sleep(1)
        print(f"weather: {path}")


def download_load(root: Path) -> None:
    destination = root / "ercot" / "load"
    destination.mkdir(parents=True, exist_ok=True)
    for year, url in LOAD_ARCHIVES.items():
        path = destination / f"load_{year}.zip"
        if not path.exists():
            path.write_bytes(_get(url).content)
        print(f"load: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--weather-end",
        default="2026-09-12",
        help="last delivery date for fixed-vintage weather (default: lockbox end)",
    )
    parser.add_argument(
        "--refresh-weather",
        action="store_true",
        help="replace existing weather JSON instead of leaving it unchanged",
    )
    args = parser.parse_args()
    download_prices(args.output)
    download_load(args.output)
    download_weather(
        args.output,
        end_date=args.weather_end,
        refresh=args.refresh_weather,
    )


if __name__ == "__main__":
    main()
