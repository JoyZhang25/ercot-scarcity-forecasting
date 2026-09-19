#!/usr/bin/env python3
"""Download the public inputs used by the case study.

The raw files remain subject to their source licenses and are excluded from Git.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile


KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024"
)
FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    "id=SP500&cosd=2008-01-01&coed=2024-12-31"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "market-news-signal/0.1 research"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    if destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded an empty file from {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="market-news-") as temporary:
        archive = Path(temporary) / "headlines.zip"
        download(KAGGLE_URL, archive)
        with ZipFile(archive) as bundle:
            member = "sp500_headlines_2008_2024.csv"
            if member not in bundle.namelist():
                raise RuntimeError(f"Expected {member}; archive contained {bundle.namelist()}")
            with bundle.open(member) as source, (args.output / member).open("wb") as target:
                shutil.copyfileobj(source, target)

    fred_path = args.output / "fred_sp500.csv"
    download(FRED_URL, fred_path)
    first_line = fred_path.read_text(encoding="utf-8").splitlines()[0]
    if first_line.strip().lower() != "observation_date,sp500":
        raise RuntimeError(f"Unexpected FRED schema: {first_line}")

    print(f"Wrote {args.output / 'sp500_headlines_2008_2024.csv'}")
    print(f"Wrote {fred_path}")


if __name__ == "__main__":
    main()
