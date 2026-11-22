from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from .metrics import best_threshold, macro_f1

def stratified_split(y, test_size=0.2, random_state=42):
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        rng.shuffle(members)
        n_test = int(round(test_size * len(members)))
        test_idx.append(members[:n_test])
        train_idx.append(members[n_test:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(test_idx))

class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        if n_splits < 2:
            raise ValueError("n_splits minimal 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        fold_of = np.empty(len(y), dtype=np.int64)
        for label in np.unique(y):
            members = np.flatnonzero(y == label)
            if self.shuffle:
                rng.shuffle(members)
            fold_of[members] = np.arange(len(members)) % self.n_splits
        for k in range(self.n_splits):
            yield np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)

@dataclass
class CVResult:
    oof_scores: np.ndarray
    macro_f1_at_half: float
    best_threshold: float
    best_macro_f1: float
    fold_macro_f1: list = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"macroF1@0.5={self.macro_f1_at_half:.4f}  "
            f"best={self.best_macro_f1:.4f}@{self.best_threshold:.2f}  "
            f"fold std={np.std(self.fold_macro_f1):.4f}  {self.seconds:.0f}s"
        )

def cross_validate(make_pipeline: Callable, make_model: Callable, df, y, n_splits=5, random_state=42, score_fn: Callable | None = None, ) -> CVResult:
    y = np.asarray(y)
    if score_fn is None:
        score_fn = lambda model, X: model.predict_proba(X)[:, 1]
    oof = np.zeros(len(y), dtype=float)
    fold_scores = []
    start = time.time()
    for train_idx, valid_idx in StratifiedKFold(n_splits, True, random_state).split(y):
        pipe = make_pipeline()
        X_train, y_train = pipe.fit_resample(df.iloc[train_idx], y[train_idx])
        X_valid = pipe.transform(df.iloc[valid_idx])
        model = make_model()
        model.fit(X_train, y_train, **getattr(pipe, "fit_kwargs", {}))
        oof[valid_idx] = score_fn(model, X_valid)
        fold_scores.append(macro_f1(y[valid_idx], (oof[valid_idx] >= 0.5).astype(int)))
    threshold, best = best_threshold(y, oof)
    return CVResult(
        oof_scores=oof,
        macro_f1_at_half=macro_f1(y, (oof >= 0.5).astype(int)),
        best_threshold=threshold,
        best_macro_f1=best,
        fold_macro_f1=fold_scores,
        seconds=time.time() - start,
    )