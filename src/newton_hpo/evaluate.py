from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from newton_hpo.config import DEFAULT_PARAMS, GLOBAL_SEED, SEARCH_SPACE


def clip_params(params: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in params.items():
        if key not in SEARCH_SPACE:
            out[key] = float(value)
            continue
        lo, hi = SEARCH_SPACE[key]
        out[key] = float(np.clip(value, lo, hi))
    return out


def merge_params(params: dict[str, float], base: dict[str, float] | None = None) -> dict[str, float]:
    merged = dict(base or DEFAULT_PARAMS)
    merged.update(params)
    return clip_params(merged)


def params_to_vector(params: dict[str, float], param_order: list[str]) -> np.ndarray:
    return np.array([params[p] for p in param_order], dtype=float)


def vector_to_params(vec: np.ndarray, param_order: list[str]) -> dict[str, float]:
    return {p: float(vec[i]) for i, p in enumerate(param_order)}


def build_xgb_pipeline(preprocess, params: dict[str, float], n_classes: int, random_state: int = 42):
    params = merge_params(params)
    max_depth = int(np.clip(round(params["max_depth"]), 3, 10))

    common = dict(
        n_estimators=200,
        max_depth=max_depth,
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
    )
    if n_classes == 2:
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    else:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            **common,
        )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def soft_tp_fp(proba, y_true, threshold, k: float = 20.0):
    """Smooth approximation of TPR / FPR (kept from the original notebook)."""
    s = 1.0 / (1.0 + np.exp(-k * (proba - threshold)))
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = np.sum(s[pos_mask]) / (np.sum(pos_mask) + 1e-8)
    fpr = np.sum(s[neg_mask]) / (np.sum(neg_mask) + 1e-8)
    return tpr, fpr


def evaluate_params(
    params: dict[str, float],
    data: dict[str, Any],
    random_state: int = GLOBAL_SEED,
    verbose: bool = False,
) -> dict[str, float]:
    params = merge_params(params)
    pipe = build_xgb_pipeline(
        preprocess=data["preprocess"],
        params=params,
        n_classes=data["n_classes"],
        random_state=random_state,
    )
    start = time.time()
    pipe.fit(data["X_train"], data["y_train"])
    elapsed = time.time() - start

    proba = pipe.predict_proba(data["X_valid"])
    pred = pipe.predict(data["X_valid"])
    acc = accuracy_score(data["y_valid"], pred)

    if data["n_classes"] == 2:
        auc = roc_auc_score(data["y_valid"], proba[:, 1])
        ll = log_loss(data["y_valid"], proba)
    else:
        auc = float("nan")
        ll = log_loss(data["y_valid"], proba)

    out = {
        "score": float(acc),
        "accuracy": float(acc),
        "logloss": float(ll),
        "auc": float(auc) if not np.isnan(auc) else float("nan"),
        "elapsed_sec": float(elapsed),
    }
    if verbose:
        print("Params:", params)
        print("Metrics:", out)
    return out


def evaluate_vector_metrics(
    params: dict[str, float],
    data: dict[str, Any],
    random_state: int = GLOBAL_SEED,
) -> np.ndarray:
    """Vector objective P(phi) = [accuracy, 1 / (1 + logloss)]."""
    metrics = evaluate_params(params, data, random_state=random_state)
    return np.array(
        [metrics["accuracy"], 1.0 / (1.0 + metrics["logloss"])],
        dtype=float,
    )


def evaluations_to_reach_fraction(history_df, fraction: float = 0.95) -> int:
    scores = history_df["score"].values
    eval_ids = np.arange(1, len(scores) + 1)
    cum_best = np.maximum.accumulate(scores)
    threshold = fraction * np.max(scores)
    idx = np.where(cum_best >= threshold)[0]
    if len(idx) == 0:
        return len(scores)
    return int(eval_ids[idx[0]])
