"""Experiment configuration with standard-library TOML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    """All choices that can affect an experiment result.

    The default execution lag is deliberately conservative. Text observed on
    feature date t is assumed to be complete only after the close. The model
    therefore enters at the close of t+1 and is evaluated on the t+1 to t+2
    close-to-close return.
    """

    train_end: str = "2019-12-31"
    validation_end: str = "2021-12-31"
    execution_lag_days: int = 1
    transaction_cost_bps: float = 5.0
    tfidf_min_df: int = 3
    tfidf_max_features: int = 12_000
    tfidf_ngram_max: int = 2
    candidate_c: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 2.0)
    bootstrap_reps: int = 1_000
    block_length: int = 10
    seed: int = 2026

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))["experiment"]
        if "candidate_c" in payload:
            payload["candidate_c"] = tuple(float(x) for x in payload["candidate_c"])
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
