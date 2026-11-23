from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import itertools
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import svm_pipeline
from src.data import development_split, load_competition_data
from src.metrics import best_threshold, macro_f1
from src.models import SVC
from src.preprocessing import TARGET
from src.validation import StratifiedKFold, stratified_split

FOLDS, Y = [], None


def prepare(n, seed):
    global FOLDS, Y
    train, _ = load_competition_data()
    dev, _ = development_split(train)
    y = dev[TARGET].to_numpy()
    _, idx = stratified_split(y, test_size=n / len(y), random_state=seed)
    sub, Y = dev.iloc[idx], y[idx]
    for tr, va in StratifiedKFold(3, True, seed).split(Y):
        pipe = svm_pipeline()
        FOLDS.append((pipe.fit_transform(sub.iloc[tr], Y[tr]), Y[tr], pipe.transform(sub.iloc[va]), va))


def run(task):
    fold, kernel, C, gamma = task
    Xtr, ytr, Xva, va = FOLDS[fold]
    params = {"kernel": kernel, "C": C, "gamma": gamma}
    if kernel == "poly":
        params.update(degree=3, coef0=1.0)
    model = SVC(**params).fit(Xtr, ytr)
    return fold, (kernel, C, gamma), model.decision_function(Xva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prepare(a.n, a.seed)
    grid = [("rbf", C, g) for C, g in itertools.product([1, 3, 10, 30], [0.03, 0.1, 0.3, 1.0])]
    grid += [("linear", C, "scale") for C in [0.1, 1]] + [("poly", C, g) for C in [0.3, 1, 3] for g in ["scale", 0.1]]
    tasks = [(f, *g) for g in grid for f in range(3)]
    oof = {}
    with get_context("fork").Pool(a.workers) as pool:
        for fold, key, f in pool.imap_unordered(run, tasks):
            oof.setdefault(key, np.zeros(len(Y)))[FOLDS[fold][3]] = f
    rows = []
    for (kernel, C, gamma), f in oof.items():
        thr, best = best_threshold(Y, f, thresholds=np.linspace(-1.5, 1.5, 121))
        rows.append({"kernel": kernel, "C": C, "gamma": gamma, "macro_f1_at_0": macro_f1(Y, (f >= 0).astype(int)),
                     "best_macro_f1": best, "best_cut": thr})
    rows.sort(key=lambda r: -r["macro_f1_at_0"])
    for r in rows:
        print(r)
    if a.out:
        Path(a.out).write_text(json.dumps({"args": vars(a), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
