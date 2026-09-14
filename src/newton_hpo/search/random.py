from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from newton_hpo.config import GLOBAL_SEED, SEARCH_SPACE
from newton_hpo.evaluate import evaluate_params


def sample_random_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "learning_rate": rng.uniform(*SEARCH_SPACE["learning_rate"]),
        "subsample": rng.uniform(*SEARCH_SPACE["subsample"]),
        "colsample_bytree": rng.uniform(*SEARCH_SPACE["colsample_bytree"]),
        "reg_lambda": float(
            np.exp(
                rng.uniform(
                    np.log(SEARCH_SPACE["reg_lambda"][0]),
                    np.log(SEARCH_SPACE["reg_lambda"][1]),
                )
            )
        ),
    }


def run_random_search(
    data: dict[str, Any],
    n_trials: int = 30,
    random_state: int = GLOBAL_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    history = []
    best_score = -float("inf")
    best_params = None
    best_metrics = None

    for i in range(n_trials):
        params = sample_random_params(rng)
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
