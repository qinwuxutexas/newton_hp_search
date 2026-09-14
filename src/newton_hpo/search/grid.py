from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import ParameterGrid

from newton_hpo.config import GLOBAL_SEED
from newton_hpo.evaluate import evaluate_params


def run_grid_search(
    data: dict[str, Any],
    random_state: int = GLOBAL_SEED,
) -> dict[str, Any]:
    param_grid = {
        "learning_rate": [0.03, 0.05, 0.08, 0.10, 0.15],
        "subsample": [0.60, 0.75, 0.90, 1.00],
        "colsample_bytree": [0.60, 0.75, 0.90, 1.00],
        "reg_lambda": [0.01, 0.10, 1.0, 5.0],
    }

    history = []
    best_score = -float("inf")
    best_params = None
    best_metrics = None

    for i, params in enumerate(ParameterGrid(param_grid)):
        metrics = evaluate_params(params, data, random_state=random_state)
        history.append({"eval_id": i, **params, **metrics})
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_params = dict(params)
            best_metrics = dict(metrics)

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "history": pd.DataFrame(history),
    }
