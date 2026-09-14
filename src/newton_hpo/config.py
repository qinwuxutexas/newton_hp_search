from __future__ import annotations

GLOBAL_SEED = 42

SEARCH_SPACE = {
    "learning_rate": (0.01, 0.30),
    "subsample": (0.50, 1.00),
    "colsample_bytree": (0.50, 1.00),
    "reg_lambda": (1e-3, 10.0),
    "reg_alpha": (0.0, 5.0),
    "max_depth": (3, 10),
    "threshold": (0.05, 0.95),
}

PARAM_ORDER = [
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "max_depth",
    "threshold",
]

INTEGER_PARAMS = {"max_depth"}
LOG_PARAMS = {"reg_lambda"}

# Four-parameter space used in the original Colab Newton runs.
NEWTON_DEFAULT_PARAM_ORDER = [
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
]

DEFAULT_PARAMS = {
    "learning_rate": 0.10,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "max_depth": 6,
    "threshold": 0.50,
}


def default_params() -> dict[str, float]:
    return dict(DEFAULT_PARAMS)
