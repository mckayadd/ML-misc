from dataclasses import dataclass
from typing import Dict, Any
from xgboost import XGBClassifier


@dataclass
class XGBConfig:
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 3
    random_state: int = 42
    eval_metric: str = "mlogloss"
    objective: str = "multi:softprob"
    num_class: int = 3


def build_model(cfg: XGBConfig) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        eval_metric=cfg.eval_metric,
        objective=cfg.objective,
        num_class=cfg.num_class
    )


def grid() -> Dict[str, list[Any]]:
    return {
        "n_estimators": [200, 400, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1]
    }