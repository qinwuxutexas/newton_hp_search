"""Damped Gauss-Newton hyperparameter search.

The original Colab notebook called ``run_newton_hpo`` and saved its history, but
the solver cell was deleted. This module rebuilds it from:

- ``evaluate_vector_metrics``: P(phi) = [accuracy, 1/(1+logloss)]
- history columns: iter, alpha, residual_norm, metrics, HPs
- ablations: ``damping_lambda``, ``phi_ref``, ``param_order``

It treats HPO as damped Gauss-Newton on ||P* - P(phi)||^2 with P* = [1, 1]
(perfect accuracy, zero logloss), optional pull toward ``phi_ref``, and a
backtracking line search starting at alpha=0.2 as in the saved run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from newton_hpo.config import (
    GLOBAL_SEED,
    INTEGER_PARAMS,
    LOG_PARAMS,
    NEWTON_DEFAULT_PARAM_ORDER,
    SEARCH_SPACE,
)
from newton_hpo.evaluate import evaluate_params, merge_params


P_TARGET = np.array([1.0, 1.0], dtype=float)


@dataclass
class NewtonHPOResult:
    best_params: dict[str, float]
    best_metrics: dict[str, float]
    history: pd.DataFrame
    n_evals: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _to_unconstrained(params: dict[str, float], param_order: list[str]) -> np.ndarray:
    z = []
    for name in param_order:
        value = float(params[name])
        if name in LOG_PARAMS:
            value = max(value, 1e-12)
            z.append(np.log(value))
        else:
            z.append(value)
    return np.array(z, dtype=float)


def _from_unconstrained(z: np.ndarray, param_order: list[str], base: dict[str, float]) -> dict[str, float]:
    params = dict(base)
    for i, name in enumerate(param_order):
        value = float(z[i])
        if name in LOG_PARAMS:
            value = float(np.exp(value))
        if name in INTEGER_PARAMS:
            value = float(int(round(value)))
        lo, hi = SEARCH_SPACE[name]
        params[name] = float(np.clip(value, lo, hi))
    return merge_params(params)


def _residual_and_metrics(
    params: dict[str, float],
    data: dict[str, Any],
    random_state: int,
    eval_fn: Callable[..., dict[str, float]],
) -> tuple[np.ndarray, dict[str, float]]:
    metrics = eval_fn(params, data, random_state=random_state)
    p = np.array(
        [metrics["accuracy"], 1.0 / (1.0 + metrics["logloss"])],
        dtype=float,
    )
    residual = P_TARGET - p
    return residual, metrics


def run_newton_hpo(
    data: dict[str, Any],
    init_params: dict[str, float] | None = None,
    n_iter: int = 8,
    random_state: int = GLOBAL_SEED,
    damping_lambda: float = 1e-2,
    phi_ref: dict[str, float] | None = None,
    param_order: list[str] | None = None,
    fd_rel_step: float = 0.02,
    init_alpha: float = 0.2,
    alpha_decay: float = 0.7,
    min_alpha: float = 0.04,
    ref_weight: float = 0.0,
    eval_fn: Callable[..., dict[str, float]] | None = None,
) -> NewtonHPOResult:
    """Run damped Gauss-Newton HPO.

    Parameters
    ----------
    damping_lambda
        Levenberg-Marquardt diagonal load on J^T J.
    phi_ref
        Optional reference hyperparameters. When set, ``ref_weight`` defaults
        to ``damping_lambda`` if you leave ``ref_weight`` at 0.
    param_order
        Which keys to optimize. Defaults to the four HPs used in the Colab run.
    """
    eval_fn = eval_fn or evaluate_params
    param_order = list(param_order or NEWTON_DEFAULT_PARAM_ORDER)
    current = merge_params(init_params or {})
    if phi_ref is not None and ref_weight == 0.0:
        ref_weight = damping_lambda
    ref_params = merge_params(phi_ref) if phi_ref is not None else dict(current)
    z_ref = _to_unconstrained(ref_params, param_order)

    z = _to_unconstrained(current, param_order)
    residual, metrics = _residual_and_metrics(current, data, random_state, eval_fn)
    n_evals = 1

    best_params = dict(current)
    best_metrics = dict(metrics)
    history_rows: list[dict[str, Any]] = []

    for it in range(n_iter):
        # Forward-difference Jacobian of P(phi) in unconstrained coordinates.
        p = P_TARGET - residual
        jac = np.zeros((2, len(param_order)), dtype=float)
        for j, name in enumerate(param_order):
            lo, hi = SEARCH_SPACE[name]
            if name in LOG_PARAMS:
                step = fd_rel_step
            elif name in INTEGER_PARAMS:
                step = 1.0
            else:
                step = fd_rel_step * (hi - lo)
            z_eps = z.copy()
            z_eps[j] += step
            params_eps = _from_unconstrained(z_eps, param_order, current)
            res_eps, _ = _residual_and_metrics(params_eps, data, random_state, eval_fn)
            p_eps = P_TARGET - res_eps
            jac[:, j] = (p_eps - p) / step
            n_evals += 1

        jtj = jac.T @ jac
        eye = np.eye(len(param_order))
        hess = jtj + damping_lambda * eye + ref_weight * eye
        grad = jac.T @ residual - ref_weight * (z - z_ref)
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hess, grad, rcond=None)[0]

        alpha = init_alpha
        accepted = False
        chosen_alpha = alpha
        chosen_z = z
        chosen_params = current
        chosen_res = residual
        chosen_metrics = metrics

        while alpha >= min_alpha:
            z_cand = z + alpha * delta
            params_cand = _from_unconstrained(z_cand, param_order, current)
            res_cand, metrics_cand = _residual_and_metrics(
                params_cand, data, random_state, eval_fn
            )
            n_evals += 1
            improved = (
                np.linalg.norm(res_cand) <= np.linalg.norm(residual) * 1.05
                or metrics_cand["score"] >= metrics["score"]
            )
            if improved:
                accepted = True
                chosen_alpha = alpha
                chosen_z = z_cand
                chosen_params = params_cand
                chosen_res = res_cand
                chosen_metrics = metrics_cand
                break
            alpha *= alpha_decay

        if not accepted:
            # Take the damped step anyway so the history length matches n_iter.
            z_cand = z + min_alpha * delta
            params_cand = _from_unconstrained(z_cand, param_order, current)
            res_cand, metrics_cand = _residual_and_metrics(
                params_cand, data, random_state, eval_fn
            )
            n_evals += 1
            chosen_alpha = min_alpha
            chosen_z = z_cand
            chosen_params = params_cand
            chosen_res = res_cand
            chosen_metrics = metrics_cand

        z = chosen_z
        current = chosen_params
        residual = chosen_res
        metrics = chosen_metrics

        row = {
            "iter": it,
            "alpha": float(chosen_alpha),
            "residual_norm": float(np.linalg.norm(residual)),
            "score": metrics["score"],
            "accuracy": metrics["accuracy"],
            "logloss": metrics["logloss"],
            "auc": metrics["auc"],
        }
        for name in param_order:
            row[name] = current[name]
        history_rows.append(row)

        if metrics["score"] > best_metrics["score"] or (
            metrics["score"] == best_metrics["score"]
            and metrics["logloss"] < best_metrics["logloss"]
        ):
            best_params = dict(current)
            best_metrics = dict(metrics)

    return NewtonHPOResult(
        best_params=best_params,
        best_metrics=best_metrics,
        history=pd.DataFrame(history_rows),
        n_evals=n_evals,
    )
