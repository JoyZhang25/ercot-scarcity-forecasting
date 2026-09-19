# Data contract

Raw files are intentionally excluded from Git. `python scripts/download_data.py`
recreates the public inputs under `data/raw/`.

| Block | Source | Role | Point-in-time treatment |
|---|---|---|---|
| RT settlement prices | ERCOT NP6-785-ER | target and lagged market state | target is the mean of four 15-minute HB_NORTH prices; predictors use lags of at least 48 hours |
| DA settlement prices | ERCOT NP4-180-ER | economic target and lagged market state | target-day DA is never supplied pre-auction; the alpha model may use the prior operating day's already-cleared curve |
| GFS forecast vintages | Open-Meteo Previous Runs API | next-day temperature stress across five Texas cities | uses `temperature_2m_previous_day2`, not realized weather |
| Native load | ERCOT Hourly Load Data Archives | retrospective demand ablation | shifted 48 hours, but not eligible for primary selection because annual archives can contain later settlement revisions |

The 2026 annual price documents are the snapshots posted on 2026-09-13; the
prospective lockbox ends on 2026-09-12. The forecast timestamp is local Central time and the delivery interval is indexed
by hour beginning. Duplicate fall-DST timestamps are averaged. The primary label
is `1{hourly RT price > $100/MWh}`; `$200/MWh` is reserved for sensitivity work.

## Why not use a current-day load forecast?

Any forecast retrieved by delivery date alone can silently select a revision posted
after the decision time. The project uses forecast-vintage weather and conservatively
lagged native load instead. ERCOT's historical load-forecast bundles are supported
only when their original posting timestamps are preserved.

## Expected local layout

```text
data/raw/
├── ercot/
│   ├── archives/{rtm,dam}_2021.zip ... {rtm,dam}_2026.zip
│   ├── load/load_2021.zip ... load_2025.zip
│   └── weather/{Austin,Dallas,Houston,Midland,SanAntonio}.json
```

Source terms remain controlling. Do not commit raw downloads or credentials.

For an existing checkout whose weather files stop in 2025, refresh them before
running the lockbox:

```bash
.venv/bin/python scripts/download_data.py \
  --weather-end 2026-09-12 \
  --refresh-weather
```
