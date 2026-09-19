"""Controlled model family spanning linear, kernel, ensemble, and neural methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit


class SeasonalPrior(BaseEstimator, ClassifierMixin):
    """Smoothed empirical spike rate by month and hour, fit on training only."""

    def __init__(self, strength: float = 48.0):
        self.strength = strength

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalPrior":
        stats = pd.DataFrame(
            {"y": np.asarray(y), "month": X.index.month, "hour": X.index.hour}
        )
        self.global_rate_ = float(np.mean(y))
        grouped = stats.groupby(["month", "hour"])["y"].agg(["sum", "count"])
        self.rates_ = (grouped["sum"] + self.strength * self.global_rate_) / (
            grouped["count"] + self.strength
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([X.index.month, X.index.hour])
        probability = self.rates_.reindex(keys).fillna(self.global_rate_).to_numpy()
        return np.column_stack([1 - probability, probability])


def candidate_models(random_state: int) -> dict[str, BaseEstimator]:
    """Return interpretable baselines plus representative ML model families."""
    def scaled(model: BaseEstimator) -> BaseEstimator:
        return make_pipeline(SimpleImputer(), StandardScaler(), model)

    return {
        "Seasonal prior": SeasonalPrior(),
        "Logistic regression": scaled(
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)
        ),
        "RBF SVM": scaled(
            CalibratedClassifierCV(
                SVC(
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=random_state,
                ),
                method="sigmoid",
                cv=TimeSeriesSplit(n_splits=3),
                ensemble=True,
            )
        ),
        "Random forest": make_pipeline(
            SimpleImputer(),
            RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=12,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "Gradient boosting": make_pipeline(
            SimpleImputer(),
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "Neural network": scaled(
            MLPClassifier(
                hidden_layer_sizes=(48, 16),
                alpha=0.01,
                batch_size=256,
                early_stopping=True,
                max_iter=300,
                random_state=random_state,
            )
        ),
    }
