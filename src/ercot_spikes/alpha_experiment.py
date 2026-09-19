"""Run the frozen walk-forward and 2026 ERCOT alpha lockbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .alpha import (
    apply_frozen_rule,
    build_alpha_frame,
    config_manifest,
    fit_and_predict,
    load_alpha_config,
    randomized_baseline_metrics,
    strategy_metrics,
)
from .alpha_plotting import plot_alpha_audit


def _period(
    frame: pd.DataFrame,
    features: list[str],
    config,
    *,
    name: str,
    fit_end: str,
    predict_start: str,
    predict_end: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    _, prediction = fit_and_predict(
        frame,
        features,
        config,
        fit_end=fit_end,
        predict_start=predict_start,
        predict_end=predict_end,
    )
    strategy = apply_frozen_rule(prediction, config)
    result = {
        "period": name,
        "fit_end": fit_end,
        "prediction_start": str(strategy.index.min()),
        "prediction_end": str(strategy.index.max()),
        **strategy_metrics(strategy, config),
    }
    return result, strategy


def run(config_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    config = load_alpha_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, features = build_alpha_frame(config)

    periods = [
        {
            "name": "2024_validation",
            "fit_end": config.initial_train_end,
            "predict_start": f"{config.validation_year}-01-01 00:00",
            "predict_end": f"{config.validation_year}-12-31 23:00",
        },
        {
            "name": "2025_shadow",
            "fit_end": f"{config.validation_year}-12-31 23:00",
            "predict_start": f"{config.shadow_year}-01-01 00:00",
            "predict_end": f"{config.shadow_year}-12-31 23:00",
        },
    ]
    if frame.index.max() >= pd.Timestamp(config.lockbox_start):
        periods.append(
            {
                "name": "2026_lockbox",
                "fit_end": f"{config.shadow_year}-12-31 23:00",
                "predict_start": config.lockbox_start,
                "predict_end": config.lockbox_end,
            }
        )

    metrics: list[dict[str, object]] = []
    baselines: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for specification in periods:
        result, strategy = _period(frame, features, config, **specification)
        metrics.append(result)
        baselines.append(
            {
                "period": specification["name"],
                **randomized_baseline_metrics(strategy, config),
            }
        )
        strategy.insert(0, "period", specification["name"])
        predictions.append(strategy)

    metrics_frame = pd.DataFrame(metrics)
    baselines_frame = pd.DataFrame(baselines)
    metrics_frame.to_csv(output / "alpha_metrics.csv", index=False)
    baselines_frame.to_csv(output / "alpha_baselines.csv", index=False)
    all_predictions = pd.concat(predictions)
    all_predictions.to_csv(
        output / "alpha_predictions.csv", index_label="timestamp"
    )
    if "2026_lockbox" in metrics_frame["period"].values:
        plot_alpha_audit(
            metrics_frame,
            all_predictions,
            Path("reports/figures/alpha_audit.png"),
        )
    lockbox_metrics = metrics_frame.loc[
        metrics_frame["period"].eq("2026_lockbox")
    ]
    alpha_gate_passed = bool(
        not lockbox_metrics.empty
        and lockbox_metrics.iloc[0]["mean_net_pnl_per_mwh"] > 0
        and lockbox_metrics.iloc[0]["mean_net_pnl_ci_95_low"] > 0
    )
    manifest: dict[str, object] = {
        "status": (
            "candidate_signal_pending_lockbox"
            if lockbox_metrics.empty
            else "alpha_gate_passed"
            if alpha_gate_passed
            else "candidate_signal_not_established"
        ),
        "alpha_gate_passed": alpha_gate_passed,
        "economic_target": "hourly HB_NORTH RT minus DA spread",
        "trade": "1 MW virtual supply in at most one hour per operating day",
        "selection_rule": (
            "choose the day's most-negative predicted RT-DA hour only when "
            f"prediction <= {config.predicted_spread_threshold:.2f} USD/MWh"
        ),
        "alpha_gate": (
            "No alpha claim unless the 2026 cost-adjusted block-bootstrap lower "
            "bound is positive and performance is not concentrated in a few days."
        ),
        "features": features,
        "config": config_manifest(config),
        "periods": metrics,
        "randomized_baselines": baselines,
    }
    with (output / "alpha_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alpha_lockbox.toml")
    parser.add_argument("--output", default="outputs/alpha")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
