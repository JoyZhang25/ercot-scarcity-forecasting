"""Interpretable baselines and text/market ablations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data import NUMERIC_FEATURES


MODEL_KINDS = ("market", "text", "combined")


@dataclass(frozen=True)
class ModelSelection:
    kind: str
    best_c: float
    validation_log_loss: float
    trials: tuple[dict[str, float | str], ...]


def _classifier(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        max_iter=2_000,
        solver="liblinear",
        random_state=seed,
    )


def make_pipeline(kind: str, c_value: float, config: ExperimentConfig) -> Pipeline:
    if kind not in MODEL_KINDS:
        raise ValueError(f"Unknown model kind: {kind}")

    if kind == "market":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", _classifier(c_value, config.seed)),
            ]
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, config.tfidf_ngram_max),
        min_df=config.tfidf_min_df,
        max_df=0.98,
        max_features=config.tfidf_max_features,
        sublinear_tf=True,
    )
    if kind == "text":
        return Pipeline(
            [
                ("tfidf", vectorizer),
                ("model", _classifier(c_value, config.seed)),
            ]
        )

    features = ColumnTransformer(
        [
            ("text", vectorizer, "headline_text"),
            ("market", StandardScaler(), list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        [
            ("features", features),
            ("model", _classifier(c_value, config.seed)),
        ]
    )


def _x(frame: pd.DataFrame, kind: str) -> pd.DataFrame | pd.Series:
    if kind == "market":
        return frame.loc[:, list(NUMERIC_FEATURES)]
    if kind == "text":
        return frame["headline_text"]
    return frame.loc[:, ["headline_text", *NUMERIC_FEATURES]]


def select_regularization(
    kind: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: ExperimentConfig,
) -> ModelSelection:
    trials: list[dict[str, float | str]] = []
    for c_value in config.candidate_c:
        pipeline = make_pipeline(kind, c_value, config)
        pipeline.fit(_x(train, kind), train["target_up"])
        probabilities = pipeline.predict_proba(_x(validation, kind))[:, 1]
        score = float(log_loss(validation["target_up"], probabilities, labels=[0, 1]))
        trials.append({"kind": kind, "c": float(c_value), "validation_log_loss": score})

    best = min(trials, key=lambda row: (float(row["validation_log_loss"]), float(row["c"])))
    return ModelSelection(
        kind=kind,
        best_c=float(best["c"]),
        validation_log_loss=float(best["validation_log_loss"]),
        trials=tuple(trials),
    )


def fit_selected(
    selection: ModelSelection,
    train_and_validation: pd.DataFrame,
    config: ExperimentConfig,
) -> Pipeline:
    pipeline = make_pipeline(selection.kind, selection.best_c, config)
    pipeline.fit(_x(train_and_validation, selection.kind), train_and_validation["target_up"])
    return pipeline


def predict_probability(model: Pipeline, frame: pd.DataFrame, kind: str) -> pd.Series:
    return pd.Series(model.predict_proba(_x(frame, kind))[:, 1], index=frame.index, name=kind)
