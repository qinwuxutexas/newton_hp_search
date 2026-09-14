from __future__ import annotations

import argparse

from newton_hpo.ablations import ablation_dimension, ablation_initialization, ablation_lambda
from newton_hpo.benchmark import aggregate_results, run_full_benchmark
from newton_hpo.config import DEFAULT_PARAMS
from newton_hpo.data import load_dataset_breast_cancer, make_splits
from newton_hpo.evaluate import evaluate_params
from newton_hpo.search import run_bayesian_optuna_tpe, run_newton_hpo, run_random_search


def smoke() -> None:
    data = make_splits(load_dataset_breast_cancer(), random_state=42)
    print(evaluate_params(DEFAULT_PARAMS, data, verbose=True))


def run_single() -> None:
    parser = argparse.ArgumentParser(description="Compare Newton HPO vs random vs TPE on breast cancer.")
    parser.add_argument("--newton-iter", type=int, default=8)
    parser.add_argument("--trials", type=int, default=20)
    args, _ = parser.parse_known_args()

    data = make_splits(load_dataset_breast_cancer(), random_state=42)
    newton = run_newton_hpo(data, DEFAULT_PARAMS, n_iter=args.newton_iter, random_state=42)
    random_res = run_random_search(data, n_trials=args.trials, random_state=42)
    bayes = run_bayesian_optuna_tpe(data, n_trials=args.trials, random_state=42)
    print("Newton:", newton.best_metrics, newton.best_params)
    print("Random:", random_res["best_metrics"], random_res["best_params"])
    print("BO/TPE:", bayes["best_metrics"], bayes["best_params"])
    print(newton.history)


def run_benchmark() -> None:
    parser = argparse.ArgumentParser(description="Multi-dataset HPO benchmark.")
    parser.add_argument("--newton-iter", type=int, default=12)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--adult", action="store_true")
    args, _ = parser.parse_known_args()

    all_results = run_full_benchmark(
        include_adult=args.adult,
        seeds=[42, 43, 44],
        run_grid=args.grid,
        n_random_trials=args.trials,
        n_bo_trials=args.trials,
        n_newton_iter=args.newton_iter,
    )
    print(aggregate_results(all_results, include_grid=args.grid))


def run_ablations() -> None:
    loader = load_dataset_breast_cancer
    print(ablation_lambda(loader))
    print(ablation_initialization(loader)[["init_id", "best_score", "best_accuracy"]])
    print(ablation_dimension(loader))
