from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import metrics

@dataclass
class FitResult:
    name: str
    pipeline: object
    model: object
    scores: np.ndarray
    threshold: float
    fit_seconds: float
    predict_seconds: float

    @property
    def predictions(self):
        return (self.scores >= self.threshold).astype(int)

def positive_scores(model, X):
    return model.predict_proba(X)[:, 1]

def fit_and_score(name, pipeline, model, train_df, y_train, eval_df, threshold=0.5, model_X=None):
    t0 = time.perf_counter()
    X_train, y_fit = pipeline.fit_resample(train_df, y_train)
    if model_X is not None:
        X_train, y_fit = model_X(X_train, y_fit)
    model.fit(X_train, y_fit, **getattr(pipeline, "fit_kwargs", {}))
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = positive_scores(model, pipeline.transform(eval_df))
    predict_seconds = time.perf_counter() - t0
    return FitResult(name, pipeline, model, scores, threshold, fit_seconds, predict_seconds)


def report_row(result: FitResult, y_true) -> dict:
    row = {"model": result.name, **metrics.classification_report(y_true, result.predictions)}
    row["threshold"] = result.threshold
    row["fit_s"] = result.fit_seconds
    row["predict_s"] = result.predict_seconds
    return row


def report_table(results, y_true) -> pd.DataFrame:
    table = pd.DataFrame([report_row(r, y_true) for r in results]).set_index("model")
    cols = ["macro_f1", "accuracy", "f1_0", "f1_1", "precision_1", "recall_1", "threshold", "fit_s", "predict_s"]
    return table[cols]


def subsample(n, random_state=42):
    from .validation import stratified_split

    def take(X, y):
        if len(y) <= n:
            return X, y
        _, idx = stratified_split(y, test_size=n / len(y), random_state=random_state)
        return X[idx], y[idx]

    return take

def segment_errors(df, y_true, y_pred, columns):
    frame = df[columns].copy()
    frame["y"] = y_true
    frame["pred"] = y_pred
    rows = []
    for column in columns:
        values = frame[column]
        if values.dtype != object and values.nunique() > 8:
            values = pd.qcut(values, 5, duplicates="drop").astype(str)
        values = values.astype(str).where(frame[column].notna(), "(missing)")
        for value, part in frame.groupby(values):
            rows.append({
                "column": column, "value": value, "n": len(part),
                "cancel_rate": part["y"].mean(),
                "error_rate": float(np.mean(part["y"] != part["pred"])),
                "macro_f1": metrics.macro_f1(part["y"], part["pred"]) if part["y"].nunique() > 1 else np.nan,
            })
    return pd.DataFrame(rows)