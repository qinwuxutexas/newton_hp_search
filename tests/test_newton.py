from __future__ import annotations

import numpy as np

from newton_hpo.config import SEARCH_SPACE
from newton_hpo.evaluate import clip_params, merge_params, params_to_vector, vector_to_params
from newton_hpo.search.newton import run_newton_hpo


def test_clip_and_roundtrip():
    params = merge_params({"learning_rate": 9.0, "subsample": 0.1})
    assert params["learning_rate"] == SEARCH_SPACE["learning_rate"][1]
    assert params["subsample"] == SEARCH_SPACE["subsample"][0]
    order = ["learning_rate", "subsample"]
    vec = params_to_vector(params, order)
    back = vector_to_params(vec, order)
    assert back["learning_rate"] == params["learning_rate"]
    assert clip_params(back)["subsample"] == params["subsample"]


def _quadratic_eval(params, data, random_state=0):
    """Smooth fake objective whose peak is near lr=0.12, subsample=0.85."""
    lr = params["learning_rate"]
    ss = params["subsample"]
    acc = 0.90 - 8.0 * (lr - 0.12) ** 2 - 4.0 * (ss - 0.85) ** 2
    acc = float(np.clip(acc, 0.5, 0.999))
    logloss = float(0.20 + 6.0 * (lr - 0.12) ** 2 + 3.0 * (ss - 0.85) ** 2)
    return {
        "score": acc,
        "accuracy": acc,
        "logloss": logloss,
        "auc": acc,
        "elapsed_sec": 0.0,
    }


def test_newton_improves_quadratic_objective():
    init = {
        "learning_rate": 0.05,
        "subsample": 0.60,
        "colsample_bytree": 0.80,
        "reg_lambda": 1.0,
    }
    start = _quadratic_eval(init, data={})
    result = run_newton_hpo(
        data={},
        init_params=init,
        n_iter=6,
        param_order=["learning_rate", "subsample"],
        eval_fn=_quadratic_eval,
        damping_lambda=1e-3,
        ref_weight=0.0,
    )
    assert result.history.shape[0] == 6
    assert "residual_norm" in result.history.columns
    assert result.best_metrics["score"] >= start["score"]
    assert abs(result.best_params["learning_rate"] - 0.12) < abs(init["learning_rate"] - 0.12)
