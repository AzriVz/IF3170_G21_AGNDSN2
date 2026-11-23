from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import itertools
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import development_split, load_competition_data
from src.metrics import best_threshold, macro_f1
from src.models import SVC
from src.preprocessing import TARGET, make_pipeline
from src.validation import StratifiedKFold, stratified_split

THRESHOLDS = np.linspace(-1.5, 1.5, 121)
FOLDS, Y = [], None

def load_dev():
    train, _ = load_competition_data()
    dev, _ = development_split(train)
    return dev, dev[TARGET].to_numpy()


def prepare(dev, y, n, n_folds, seed):
    global FOLDS, Y
    FOLDS = []
    if 0 < n < len(y):
        _, idx = stratified_split(y, test_size=n / len(y), random_state=seed)
        sub, Y = dev.iloc[idx], y[idx]
    else:
        sub, Y = dev, y
    for tr, va in StratifiedKFold(n_folds, True, seed).split(Y):
        pipe = make_pipeline()
        FOLDS.append((pipe.fit_transform(sub.iloc[tr], Y[tr]), Y[tr], pipe.transform(sub.iloc[va]), va))

def build_grid():
    cws = [None, "balanced"]
    grid = []
    for C, g, cw in itertools.product([0.3, 1, 3, 10], [0.003, 0.01, 0.03, 0.1], cws):
        grid.append(("rbf", C, g, 3, 0.0, cw))
    for C, cw in itertools.product([0.01, 0.03, 0.1, 0.3, 1], cws):
        grid.append(("linear", C, "scale", 3, 0.0, cw))
    for C, deg, c0, cw in itertools.product([0.3, 1, 3], [2, 3], [0.0, 1.0], cws):
        grid.append(("poly", C, "scale", deg, c0, cw))
    return grid


def run(task):
    fold, cfg = task
    kernel, C, gamma, degree, coef0, cw = cfg
    Xtr, ytr, Xva, _ = FOLDS[fold]
    params = {"kernel": kernel, "C": C, "gamma": gamma, "class_weight": cw}
    if kernel == "poly":
        params.update(degree=degree, coef0=coef0)
    model = SVC(**params).fit(Xtr, ytr)
    return fold, cfg, model.decision_function(Xva)

def cv_threshold_f1(f):
    pred = np.zeros(len(Y), dtype=int)
    for _, _, _, va in FOLDS:
        rest = np.ones(len(Y), dtype=bool)
        rest[va] = False
        thr, _ = best_threshold(Y[rest], f[rest], thresholds=THRESHOLDS)
        pred[va] = (f[va] >= thr).astype(int)
    return macro_f1(Y, pred)


def evaluate(grid, workers):
    tasks = [(f, cfg) for cfg in grid for f in range(len(FOLDS))]
    oof = {}
    with get_context("fork").Pool(workers) as pool:
        for fold, cfg, f in pool.imap_unordered(run, tasks):
            oof.setdefault(cfg, np.zeros(len(Y)))[FOLDS[fold][3]] = f
    rows = []
    for cfg, f in oof.items():
        kernel, C, gamma, degree, coef0, cw = cfg
        thr, best = best_threshold(Y, f, thresholds=THRESHOLDS)
        rows.append({
            "kernel": kernel, "C": C, "gamma": gamma, "degree": degree, "coef0": coef0,
            "class_weight": cw,
            "macro_f1_at_0": macro_f1(Y, (f >= 0).astype(int)),
            "best_macro_f1": best,
            "best_cut": float(thr),
            "cv_thr_macro_f1": cv_threshold_f1(f),
        })
    rows.sort(key=lambda r: -r["cv_thr_macro_f1"])
    return rows


def warn_edges(rows, grid):
    names = ["kernel", "C", "gamma", "degree", "coef0", "class_weight"]
    for kernel in sorted({g[0] for g in grid}):
        best = next((r for r in rows if r["kernel"] == kernel), None)
        if best is None:
            continue
        sub = [g for g in grid if g[0] == kernel]
        for i, name in enumerate(names[1:], start=1):
            vals = {g[i] for g in sub if not isinstance(g[i], str) and g[i] is not None}
            if len(vals) < 2 or isinstance(best[name], str) or best[name] is None:
                continue
            if best[name] in (min(vals), max(vals)):
                side = "min" if best[name] == min(vals) else "max"
                print(f"[EDGE] {kernel}: best {name}={best[name]} is the {side} of the tested range "
                      f"-> extend the grid in that direction")


def show(title, rows, top=None):
    print(f"\n=== {title} ===")
    for r in rows[:top]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="stage-1 subsample size")
    ap.add_argument("--folds1", type=int, default=3, help="stage-1 CV folds")
    ap.add_argument("--folds2", type=int, default=5, help="stage-2 CV folds (full dev set)")
    ap.add_argument("--top", type=int, default=3, help="configs re-checked in stage 2 (0 = skip stage 2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev, y = load_dev()
    grid = build_grid()

    t0 = time.time()
    prepare(dev, y, a.n, a.folds1, a.seed)
    stage1 = evaluate(grid, a.workers)
    print(f"stage 1: {len(grid)} configs, n={len(Y)}, {a.folds1}-fold, {time.time() - t0:.0f}s")
    show("STAGE 1 (ranked by cv_thr_macro_f1)", stage1)
    warn_edges(stage1, grid)

    stage2 = []
    if a.top > 0:
        keys = ["kernel", "C", "gamma", "degree", "coef0", "class_weight"]
        top_cfgs = [tuple(r[k] for k in keys) for r in stage1[:a.top]]
        t0 = time.time()
        prepare(dev, y, 0, a.folds2, a.seed)
        stage2 = evaluate(top_cfgs, a.workers)
        print(f"\nstage 2: {len(top_cfgs)} configs, n={len(Y)}, {a.folds2}-fold, {time.time() - t0:.0f}s")
        show("STAGE 2 - full dev set (ranked by cv_thr_macro_f1)", stage2)

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"args": vars(a), "stage1": stage1, "stage2": stage2}, indent=2, default=str))


if __name__ == "__main__":
    main()