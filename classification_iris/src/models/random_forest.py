from dataclasses import dataclass
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier

@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: int | None = 5
    max_features: int | str | None = 2
    random_state: int = 42


def build_model(cfg: RFConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        max_features=cfg.max_features,
        random_state=cfg.random_state
    )


def grid() -> Dict[str, list[Any]]:
    return {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, None],
        "max_features": [2, 3, 4]
    }