"""Run the frozen chronological ERCOT spike-forecast experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression

from .config import load_config
from .data import build_feature_frame, chronological_split
from .evaluation import classification_metrics, daily_block_interval, risk_bins
from .models import candidate_models
from .plotting import (
    plot_calibration,
    plot_feature_importance,
    plot_model_comparison,
    plot_risk_lift,
    plot_timeline,
)


def run(config_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    frame, features = build_feature_frame(config)
    train, validation, test = chronological_split(frame, config)
    features = [feature for feature in features if train[feature].notna().any()]
    core_features = [
        feature
        for feature in features
        if not feature.startswith("load_") and not feature.startswith("actual_load_")
    ]
    y_train = train["spike"]
    y_validation = validation["spike"]
    y_test = test["spike"]

    models = candidate_models(config.random_state)
    model_specs = {name: (model, core_features) for name, model in models.items()}
    if len(core_features) < len(features):
        load_model = candidate_models(config.random_state)["Gradient boosting"]
        model_specs["Gradient boosting + lagged load"] = (load_model, features)
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    fitted = {}
    for name, (model, model_features) in model_specs.items():
        model.fit(train[model_features], y_train)
        fitted[name] = (model, model_features)
        validation_probability = model.predict_proba(validation[model_features])[:, 1]
        test_probability = model.predict_proba(test[model_features])[:, 1]
        validation_rows.append(
            {"model": name, **classification_metrics(y_validation, validation_probability, config.top_risk_fraction)}
        )
        test_rows.append(
            {"model": name, **classification_metrics(y_test, test_probability, config.top_risk_fraction)}
        )

    validation_metrics = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False)
    test_metrics = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    eligible = validation_metrics.loc[
        ~validation_metrics["model"].str.contains("lagged load", case=False)
    ]
    selected_name = str(eligible.iloc[0]["model"])
    selected, selected_features = fitted[selected_name]

    validation_raw = selected.predict_proba(validation[selected_features])[:, 1]
    test_raw = selected.predict_proba(test[selected_features])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    calibrator.fit(validation_raw, y_validation)
    probability = calibrator.predict(test_raw)
    calibrated_metrics = classification_metrics(y_test, probability, config.top_risk_fraction)

    predictions = test[["rtm_price", "dam_price", "spike"]].copy()
    predictions["probability_raw"] = test_raw
    predictions["probability"] = probability
    predictions["spread"] = predictions["rtm_price"] - predictions["dam_price"]
    bins = risk_bins(predictions.dropna(subset=["spread"]))
    spread_estimate, spread_low, spread_high = daily_block_interval(
        predictions.dropna(subset=["spread"]), seed=config.random_state
    )

    permutation = permutation_importance(
        selected,
        test[selected_features],
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=config.random_state,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": selected_features,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    validation_metrics.to_csv(output / "validation_metrics.csv", index=False)
    test_metrics.to_csv(output / "test_metrics.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index_label="timestamp")
    bins.to_csv(output / "risk_deciles.csv", index=False)
    importance.to_csv(output / "permutation_importance.csv", index=False)

    plot_timeline(predictions, figures / "scarcity_timeline.png")
    plot_model_comparison(test_metrics, figures / "model_comparison.png")
    plot_risk_lift(bins, figures / "risk_lift.png")
    plot_feature_importance(importance, figures / "feature_importance.png")
    plot_calibration(y_test, probability, figures / "calibration.png")

    manifest: dict[str, object] = {
        "research_question": "Can public information available before the day-ahead auction identify next-day HB_NORTH real-time price spikes?",
        "target": f"hourly RT price > ${config.spike_threshold:.0f}/MWh",
        "model_selection": {
            "period": "2024",
            "metric": "precision-recall AUC",
            "eligible_models": eligible["model"].tolist(),
            "selection_ineligible_diagnostics": [
                name for name in validation_metrics["model"] if name not in set(eligible["model"])
            ],
        },
        "selected_on_2024_validation": selected_name,
        "selected_features": selected_features,
        "test_period": [str(test.index.min()), str(test.index.max())],
        "test_observations": int(len(test)),
        "test_spikes": int(y_test.sum()),
        "raw_selected_model_test_metrics": {
            key: float(value)
            for key, value in test_metrics.loc[test_metrics["model"] == selected_name]
            .drop(columns="model")
            .iloc[0]
            .items()
        },
        "calibrated_test_metrics": {key: float(value) for key, value in calibrated_metrics.items()},
        "top_decile_spread_lift": {
            "estimate": spread_estimate,
            "daily_block_bootstrap_95pct": [spread_low, spread_high],
        },
        "timing_guardrail": f"realized power-system variables are lagged at least {config.minimum_lag_hours} hours",
        "random_state": config.random_state,
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.toml")
    parser.add_argument("--output", default="outputs/benchmark")
    args = parser.parse_args()
    manifest = run(args.config, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
