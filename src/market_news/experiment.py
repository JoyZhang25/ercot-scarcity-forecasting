"""End-to-end benchmark runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn

from .config import ExperimentConfig
from .data import build_daily_panel, load_headlines, load_market, temporal_split
from .evaluation import (
    classification_metrics,
    log_loss_differential,
    newey_west_mean_test,
    paired_block_bootstrap_log_loss,
    strategy_metrics,
)
from .models import MODEL_KINDS, fit_selected, predict_probability, select_regularization
from .plotting import plot_diagnostics, plot_model_comparison


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiment(
    *,
    headlines_path: str | Path,
    market_path: str | Path,
    output_dir: str | Path,
    config: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    headlines = load_headlines(headlines_path)
    market = load_market(market_path)
    panel = build_daily_panel(
        headlines,
        market,
        execution_lag_days=config.execution_lag_days,
    )
    split = temporal_split(
        panel,
        train_end=config.train_end,
        validation_end=config.validation_end,
    )
    train_and_validation = pd.concat([split.train, split.validation], ignore_index=True)

    selection_rows: list[dict[str, float | str]] = []
    probabilities: dict[str, np.ndarray] = {}
    for kind in MODEL_KINDS:
        selection = select_regularization(kind, split.train, split.validation, config)
        selection_rows.extend(selection.trials)
        model = fit_selected(selection, train_and_validation, config)
        probabilities[kind] = predict_probability(model, split.test, kind).to_numpy()

    prior = float(train_and_validation["target_up"].mean())
    probabilities["prior"] = np.full(len(split.test), prior, dtype=float)

    y_test = split.test["target_up"].to_numpy()
    target_return = split.test["target_return"].to_numpy()
    metric_rows = []
    strategy_rows = []
    for model_name in ("prior", *MODEL_KINDS):
        metric_rows.append(
            {"model": model_name, **classification_metrics(y_test, probabilities[model_name])}
        )
        strategy_rows.append(
            {
                "model": model_name,
                **strategy_metrics(
                    probabilities[model_name],
                    target_return,
                    transaction_cost_bps=config.transaction_cost_bps,
                ),
            }
        )

    comparison_rows = []
    comparison_pairs = (
        ("prior", "market"),
        ("prior", "text"),
        ("prior", "combined"),
        ("market", "combined"),
    )
    for baseline, candidate in comparison_pairs:
        bootstrap = paired_block_bootstrap_log_loss(
            y_test,
            probabilities[baseline],
            probabilities[candidate],
            reps=config.bootstrap_reps,
            block_length=config.block_length,
            seed=config.seed,
        )
        differential = log_loss_differential(
            y_test,
            probabilities[baseline],
            probabilities[candidate],
        )
        hac = newey_west_mean_test(differential, max_lag=config.block_length - 1)
        comparison_rows.append(
            {"baseline": baseline, "candidate": candidate, **bootstrap, **hac}
        )

    metrics = pd.DataFrame(metric_rows)
    strategies = pd.DataFrame(strategy_rows)
    selections = pd.DataFrame(selection_rows)
    comparisons = pd.DataFrame(comparison_rows)
    predictions = split.test[
        ["feature_date", "execution_date", "outcome_date", "target_return", "target_up"]
    ].copy()
    for model_name, values in probabilities.items():
        predictions[f"p_{model_name}"] = values

    metrics.to_csv(output / "classification_metrics.csv", index=False)
    strategies.to_csv(output / "strategy_metrics.csv", index=False)
    selections.to_csv(output / "model_selection.csv", index=False)
    comparisons.to_csv(output / "model_comparisons.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    plot_model_comparison(metrics, output / "model_performance.png")
    plot_diagnostics(
        predictions,
        output / "diagnostics.png",
        transaction_cost_bps=config.transaction_cost_bps,
    )

    manifest = {
        "config": config.to_dict(),
        "input_hashes": {
            "headlines_sha256": _sha256(headlines_path),
            "market_sha256": _sha256(market_path),
        },
        "sample_counts": {
            "headlines": len(headlines),
            "market_sessions": len(market),
            "panel": len(panel),
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "date_ranges": {
            "train": [str(split.train.feature_date.min().date()), str(split.train.outcome_date.max().date())],
            "validation": [
                str(split.validation.feature_date.min().date()),
                str(split.validation.outcome_date.max().date()),
            ],
            "test": [str(split.test.feature_date.min().date()), str(split.test.outcome_date.max().date())],
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "strategies": strategies,
        "selections": selections,
        "comparisons": comparisons,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headlines", required=True, help="CSV with dated financial headlines")
    parser.add_argument("--market", required=True, help="CSV with daily S&P 500 closes")
    parser.add_argument("--config", default="configs/benchmark.toml")
    parser.add_argument("--output", default="outputs/benchmark")
    args = parser.parse_args()
    config = ExperimentConfig.from_toml(args.config)
    results = run_experiment(
        headlines_path=args.headlines,
        market_path=args.market,
        output_dir=args.output,
        config=config,
    )
    print(results["metrics"].to_string(index=False))
    print("\nMarket-relative loss tests")
    print(results["comparisons"].to_string(index=False))


if __name__ == "__main__":
    main()
