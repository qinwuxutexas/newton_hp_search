from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_wine
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from newton_hpo.config import GLOBAL_SEED


def _make_numeric_bundle_from_sklearn_dataset(dataset, name: str) -> dict[str, Any]:
    X = pd.DataFrame(dataset.data, columns=getattr(dataset, "feature_names", None))
    if X.columns.isnull().any():
        X.columns = [f"f{i}" for i in range(X.shape[1])]
    y = pd.Series(dataset.target, name="target")
    return {
        "name": name,
        "X": X,
        "y": y,
        "categorical_cols": [],
        "numerical_cols": X.columns.tolist(),
    }


def load_dataset_breast_cancer() -> dict[str, Any]:
    return _make_numeric_bundle_from_sklearn_dataset(load_breast_cancer(), "breast_cancer")


def load_dataset_wine() -> dict[str, Any]:
    return _make_numeric_bundle_from_sklearn_dataset(load_wine(), "wine")


def load_dataset_digits() -> dict[str, Any]:
    return _make_numeric_bundle_from_sklearn_dataset(load_digits(), "digits")


def load_dataset_adult_openml() -> dict[str, Any]:
    X, y = fetch_openml(name="adult", version=2, as_frame=True, return_X_y=True)
    y = pd.Series((y.astype(str) == ">50K").astype(int), name="target")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    return {
        "name": "adult_openml",
        "X": X,
        "y": y,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
    }


def get_public_datasets(include_adult: bool = False) -> list[dict[str, Any]]:
    datasets = [
        load_dataset_breast_cancer(),
        load_dataset_wine(),
        load_dataset_digits(),
    ]
    if include_adult:
        try:
            datasets.append(load_dataset_adult_openml())
        except Exception as exc:
            print(f"Skipping Adult/OpenML due to error: {exc}")
    return datasets


def make_splits(
    dataset_bundle: dict[str, Any],
    random_state: int = GLOBAL_SEED,
    test_size: float = 0.20,
    valid_size_within_train: float = 0.25,
) -> dict[str, Any]:
    X = dataset_bundle["X"]
    y = dataset_bundle["y"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=valid_size_within_train,
        random_state=random_state,
        stratify=y_train_full,
    )

    categorical_cols = dataset_bundle["categorical_cols"]
    numerical_cols = dataset_bundle["numerical_cols"]

    if categorical_cols:
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    categorical_cols,
                ),
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numerical_cols,
                ),
            ],
            remainder="drop",
        )
    else:
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numerical_cols,
                ),
            ],
            remainder="drop",
        )

    return {
        "name": dataset_bundle["name"],
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "preprocess": preprocess,
        "n_classes": int(y.nunique()),
    }
