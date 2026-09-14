from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from newton_hpo.config import DEFAULT_PARAMS
from newton_hpo.data import make_splits
from newton_hpo.search.newton import run_newton_hpo


def ablation_lambda(
    dataset_loader: Callable[[], dict[str, Any]],
    seed: int = 42,
    lambda_values: list[float] | None = None,
    n_iter: int = 12,
) -> pd.DataFrame:
    lambda_values = lambda_values or [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    data = make_splits(dataset_loader(), random_state=seed)
    rows = []
    for lam in lambda_values:
        res = run_newton_hpo(
            data=data,
            init_params=DEFAULT_PARAMS,
            damping_lambda=lam,
            n_iter=n_iter,
            random_state=seed,
        )
        rows.append({
            "lambda": lam,
            "best_score": res.best_metrics["score"],
            "best_accuracy": res.best_metrics["accuracy"],
            "best_logloss": res.best_metrics["logloss"],
        })
    return pd.DataFrame(rows)


def ablation_initialization(
    dataset_loader: Callable[[], dict[str, Any]],
    seed: int = 42,
    n_iter: int = 12,
) -> pd.DataFrame:
    data = make_splits(dataset_loader(), random_state=seed)
    init_list = [
        {"learning_rate": 0.03, "subsample": 0.60, "colsample_bytree": 0.60, "reg_lambda": 0.01},
        {"learning_rate": 0.10, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 1.00},
        {"learning_rate": 0.20, "subsample": 0.95, "colsample_bytree": 0.95, "reg_lambda": 5.00},
    ]
    rows = []
    for i, init_params in enumerate(init_list):
        res = run_newton_hpo(
            data=data,
            init_params=init_params,
            phi_ref=init_params,
            n_iter=n_iter,
            random_state=seed,
        )
        rows.append({
            "init_id": i,
            "init_params": init_params,
            "best_score": res.best_metrics["score"],
            "best_accuracy": res.best_metrics["accuracy"],
            "best_logloss": res.best_metrics["logloss"],
            "best_params": res.best_params,
        })
    return pd.DataFrame(rows)


def ablation_dimension(
    dataset_loader: Callable[[], dict[str, Any]],
    seed: int = 42,
    n_iter: int = 12,
) -> pd.DataFrame:
    data = make_splits(dataset_loader(), random_state=seed)
    param_sets = [
        ["learning_rate", "subsample"],
        ["learning_rate", "subsample", "colsample_bytree"],
        ["learning_rate", "subsample", "colsample_bytree", "reg_lambda"],
    ]
    rows = []
    for param_order in param_sets:
        res = run_newton_hpo(
            data=data,
            init_params=DEFAULT_PARAMS,
            param_order=param_order,
            n_iter=n_iter,
            random_state=seed,
        )
        rows.append({
            "n_params": len(param_order),
            "param_order": tuple(param_order),
            "best_score": res.best_metrics["score"],
            "best_accuracy": res.best_metrics["accuracy"],
            "best_logloss": res.best_metrics["logloss"],
        })
    return pd.DataFrame(rows)
