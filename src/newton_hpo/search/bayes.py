from __future__ import annotations

from typing import Any

import pandas as pd

from newton_hpo.config import GLOBAL_SEED
from newton_hpo.evaluate import evaluate_params


def run_bayesian_optuna_tpe(
    data: dict[str, Any],
    n_trials: int = 30,
    random_state: int = GLOBAL_SEED,
) -> dict[str, Any]:
    import optuna
    from optuna.samplers import TPESampler

    history: list[dict[str, Any]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30),
            "subsample": trial.suggest_float("subsample", 0.50, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        metrics = evaluate_params(params, data, random_state=random_state)
        history.append({"eval_id": trial.number, **params, **metrics})
        return metrics["score"]

    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_metrics = evaluate_params(best_params, data, random_state=random_state)
    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "history": pd.DataFrame(history),
        "study": study,
    }
