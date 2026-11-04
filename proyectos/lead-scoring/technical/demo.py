"""Synthetic-only CLI and reusable training-frame nested evaluation."""
import argparse

import json

import math

import warnings

from pathlib import Path

import numpy as np

import pandas as pd

import sklearn

from sklearn.compose import ColumnTransformer

from sklearn.exceptions import ConvergenceWarning

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

FORM = ["origen", "fuente", "ambito", "ocupacion"]

NUMERIC = ["visitas_total", "tiempo_en_site_total", "paginas_vistas_visita",
           "score_actividad", "score_perfil"]

def ranking_metrics(y, scores):
    """Top ceil(10% n), ties resolved by input source order, never by target."""
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Nonempty aligned one-dimensional inputs required")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Two binary classes and finite scores required")
    k = math.ceil(0.10 * len(y))
    selected = np.argsort(-scores, kind="stable")[:k]
    positives = int(y[selected].sum())
    return {"n": len(y), "k": k, "positives_top_k": positives,
            "prevalence": float(y.mean()), "roc_auc": float(roc_auc_score(y, scores)),
            "precision_at_10pct": positives / k,
            "capture_at_10pct": positives / int(y.sum()),
            "lift_at_10pct": float((positives / k) / y.mean())}

def build_pipeline(expanded):
    categories = FORM + (["ult_actividad"] if expanded else [])
    category = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                         ("encode", OneHotEncoder(handle_unknown="ignore"))])
    transformers = [("category", category, categories)]
    if expanded:
        numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                     keep_empty_features=True)),
                            ("scale", MinMaxScaler())])
        transformers.append(("numeric", numeric, NUMERIC))
    return Pipeline([("prepare", ColumnTransformer(transformers)),
                     ("model", LogisticRegression(solver="saga", l1_ratio=1.0,
                                                  max_iter=5000, random_state=42))])


def evaluate(frame):
    """Accept a training-only frame, preserve identity and return aggregates only."""
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Unique source indices and columns required")
    y = frame.compra.to_numpy()
    if set(np.unique(y)) != {0, 1} or min(np.bincount(y)) < 10:
        raise ValueError("Binary target with at least ten examples per class required")
    outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(frame, y))
    results = {"scenarios": {}, "fold_metrics": [],
               "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                            "numpy": np.__version__}}
    for name, expanded in (("form", False), ("expanded_unverified", True)):
        columns = FORM + (["ult_actividad"] + NUMERIC if expanded else [])
        X = frame.loc[:, columns].copy()
        for column in FORM + (["ult_actividad"] if expanded else []):
            X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
        oof = np.full(len(frame), np.nan)
        for fold, (fit, test) in enumerate(outer):
            search = GridSearchCV(build_pipeline(expanded), {"model__C": [0.1, 1.0, 10.0]},
                                  scoring="roc_auc", n_jobs=1, error_score="raise",
                                  cv=StratifiedKFold(3, shuffle=True, random_state=42))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                search.fit(X.iloc[fit], y[fit])
            oof[test] = search.predict_proba(X.iloc[test])[:, 1]
            results["fold_metrics"].append({"scenario": name, "fold": fold,
                                           "selected_C": search.best_params_["model__C"],
                                           **ranking_metrics(y[test], oof[test])})
        results["scenarios"][name] = {**ranking_metrics(y, oof), "coverage": 1.0}
    return results


def write_results(path, results):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)


if __name__ == "__main__":
    from generate_synthetic_data import generate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New JSON file for synthetic aggregate metrics")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists, choose a new path")
    result = evaluate(generate())
    result["data_origin"] = "synthetic_only_not_real_performance"
    write_results(args.output, result)
    print("Synthetic demonstration completed, these scores are not real-data evidence.")
