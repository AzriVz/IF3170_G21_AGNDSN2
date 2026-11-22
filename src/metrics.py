from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    return skm.confusion_matrix(y_true, y_pred, labels=labels)


def precision_recall_f1(y_true, y_pred, labels=None):
    precision, recall, f1, _ = skm.precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return precision, recall, f1


def accuracy(y_true, y_pred) -> float:
    return float(skm.accuracy_score(y_true, y_pred))

def f1_score(y_true, y_pred, positive=1) -> float:
    return float(skm.f1_score(y_true, y_pred, pos_label=positive, average="binary", zero_division=0))

def macro_f1(y_true, y_pred, labels=(0, 1)) -> float:
    return float(skm.f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))

def macro_f1_curve(y_true, scores, thresholds) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return np.array([macro_f1(y_true, (scores >= t).astype(int)) for t in np.asarray(thresholds, dtype=float)])

def best_threshold(y_true, scores, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 4)
    thresholds = np.asarray(thresholds, dtype=float)
    curve = macro_f1_curve(y_true, scores, thresholds)
    best = curve.max()
    ties = np.flatnonzero(np.isclose(curve, best, rtol=0, atol=1e-12))
    chosen = ties[len(ties) // 2]
    return float(thresholds[chosen]), float(best)

def classification_report(y_true, y_pred, labels=(0, 1)) -> dict:
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, labels=list(labels))
    report = {"accuracy": accuracy(y_true, y_pred), "macro_f1": float(f1.mean())}
    for label, p, r, f in zip(labels, precision, recall, f1):
        report[f"precision_{label}"] = float(p)
        report[f"recall_{label}"] = float(r)
        report[f"f1_{label}"] = float(f)
    return report

def misclassification_cost(y_true, y_pred, cost_fp=2.0, cost_fn=1.0) -> float:
    (_, fp), (fn, _) = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    return float(cost_fp * fp + cost_fn * fn)