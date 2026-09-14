from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from newton_hpo.config import DEFAULT_PARAMS, PARAM_ORDER
from newton_hpo.data import (
    load_dataset_adult_openml,
    load_dataset_breast_cancer,
    load_dataset_digits,
    load_dataset_wine,
    make_splits,
)
from newton_hpo.evaluate import evaluations_to_reach_fraction
from newton_hpo.search import (
    run_bayesian_optuna_tpe,
    run_grid_search,
    run_newton_hpo,
    run_random_search,
)


def benchmark_one_dataset_one_seed(
    dataset_loader: Callable[[], dict[str, Any]],
    seed: int,
    run_grid: bool = False,
    n_random_trials: int = 30,
    n_bo_trials: int = 30,
    n_newton_iter: int = 12,
) -> tuple[str, dict[str, Any]]:
    dataset_bundle = dataset_loader()
    data = make_splits(dataset_bundle, random_state=seed)
    out: dict[str, Any] = {}

    newton_res = run_newton_hpo(
        data=data,
        init_params=DEFAULT_PARAMS,
        n_iter=n_newton_iter,
        random_state=seed,
    )
    out["newton"] = {
        "best_metrics": newton_res.best_metrics,
        "best_params": newton_res.best_params,
        "history": newton_res.history.copy(),
        "n_evals": newton_res.n_evals,
    }

    out["random"] = run_random_search(
        data=data, n_trials=n_random_trials, random_state=seed,
    )
    out["bayes"] = run_bayesian_optuna_tpe(
        data=data, n_trials=n_bo_trials, random_state=seed,
    )
    if run_grid:
        out["grid"] = run_grid_search(data=data, random_state=seed)
    return dataset_bundle["name"], out


def run_full_benchmark(
    include_adult: bool = False,
    seeds: list[int] | None = None,
    run_grid: bool = False,
    n_random_trials: int = 30,
    n_bo_trials: int = 30,
    n_newton_iter: int = 12,
) -> dict[str, Any]:
    seeds = seeds or [42, 43, 44]
    dataset_loaders = [
        load_dataset_breast_cancer,
        load_dataset_wine,
        load_dataset_digits,
    ]
    if include_adult:
        dataset_loaders.append(load_dataset_adult_openml)

    all_results: dict[str, Any] = {}
    for loader in dataset_loaders:
        dataset_name = loader()["name"]
        all_results[dataset_name] = {}
        for seed in seeds:
            print(f"Running dataset={dataset_name}, seed={seed}")
            name, result = benchmark_one_dataset_one_seed(
                dataset_loader=loader,
                seed=seed,
                run_grid=run_grid,
                n_random_trials=n_random_trials,
                n_bo_trials=n_bo_trials,
                n_newton_iter=n_newton_iter,
            )
            all_results[name][seed] = result
    return all_results


def aggregate_results(all_results: dict[str, Any], include_grid: bool = False) -> pd.DataFrame:
    rows = []
    methods = ["newton", "random", "bayes"] + (["grid"] if include_grid else [])
    for dataset_name, per_seed in all_results.items():
        for method in methods:
            best_scores = []
            accuracies = []
            evals_95 = []
            for _seed, result in per_seed.items():
                if method not in result:
                    continue
                history = result[method]["history"]
                best_metrics = result[method]["best_metrics"]
                best_scores.append(best_metrics["score"])
                accuracies.append(best_metrics["accuracy"])
                evals_95.append(evaluations_to_reach_fraction(history, fraction=0.95))
            rows.append({
                "dataset": dataset_name,
                "method": method,
                "best_score_mean": float(np.mean(best_scores)),
                "best_score_std": float(np.std(best_scores)),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "evals95_mean": float(np.mean(evals_95)),
                "evals95_std": float(np.std(evals_95)),
            })
    return pd.DataFrame(rows)


def build_example_hparam_table(
    all_results: dict[str, Any],
    dataset_name: str,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for method in ["bayes", "newton"]:
        best_params = all_results[dataset_name][seed][method]["best_params"]
        row = {"method": method}
        for p in PARAM_ORDER:
            row[p] = best_params.get(p)
        rows.append(row)
    return pd.DataFrame(rows)
