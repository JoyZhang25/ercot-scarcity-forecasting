import pandas as pd

from ercot_spikes.models import SeasonalPrior, candidate_models


def test_seasonal_prior_is_fit_on_training_labels() -> None:
    index = pd.date_range("2023-01-01", periods=96, freq="h")
    X = pd.DataFrame({"feature": range(96)}, index=index)
    y = pd.Series(([0] * 72) + ([1] * 24), index=index)
    model = SeasonalPrior(strength=10).fit(X, y)
    probability = model.predict_proba(X)
    assert probability.shape == (96, 2)
    assert ((probability >= 0) & (probability <= 1)).all()


def test_model_family_covers_core_ml_techniques() -> None:
    models = candidate_models(7641)
    assert set(models) == {
        "Seasonal prior",
        "Logistic regression",
        "RBF SVM",
        "Random forest",
        "Gradient boosting",
        "Neural network",
    }
