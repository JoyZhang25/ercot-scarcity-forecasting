import pandas as pd

from market_news.config import ExperimentConfig
from market_news.models import make_pipeline


def test_text_vocabulary_is_fit_on_training_data_only() -> None:
    train = pd.DataFrame(
        {
            "headline_text": [
                "profits expand strongly",
                "losses deepen sharply",
                "profits beat expectations",
                "losses miss estimates",
            ],
            "target_up": [1, 0, 1, 0],
        }
    )
    validation = pd.DataFrame(
        {"headline_text": ["futureleaktoken appears"], "target_up": [1]}
    )
    config = ExperimentConfig(tfidf_min_df=1, candidate_c=(1.0,))
    model = make_pipeline("text", 1.0, config)
    model.fit(train["headline_text"], train["target_up"])
    vocabulary = model.named_steps["tfidf"].vocabulary_
    assert "futureleaktoken" not in vocabulary
    assert model.predict_proba(validation["headline_text"]).shape == (1, 2)
