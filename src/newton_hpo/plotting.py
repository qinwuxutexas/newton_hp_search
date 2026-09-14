from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence_one_dataset_seed(
    all_results: dict[str, Any],
    dataset_name: str,
    seed: int = 42,
    include_grid: bool = False,
    show: bool = True,
    save_path: str | None = None,
):
    plt.figure(figsize=(8, 5))
    methods = ["newton", "random", "bayes"] + (["grid"] if include_grid else [])
    labels = {
        "newton": "Proposed",
        "random": "Random Search",
        "bayes": "Bayesian Opt.",
        "grid": "Grid Search",
    }
    for method in methods:
        hist = all_results[dataset_name][seed][method]["history"].copy()
        y = np.maximum.accumulate(hist["score"].values)
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=labels[method])
    plt.xlabel("Evaluation count")
    plt.ylabel("Best validation score so far")
    plt.title(f"Convergence: {dataset_name}, seed={seed}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
