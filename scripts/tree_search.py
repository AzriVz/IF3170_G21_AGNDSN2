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
from src.models import CARTClassifier
from src.preprocessing import TARGET, make_pipeline
from src.validation import StratifiedKFold

FOLDS, Y = [], None
ALPHAS = [0.0, 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]

def prepare(impute, derived, seed):
    global FOLDS, Y
    train, _ = load_competition_data()
    dev, _ = development_split(train)
    Y = dev[TARGET].to_numpy()
    FOLDS = []
    for tr, va in StratifiedKFold(5, True, seed).split(Y):
        pipe = make_pipeline("tree", impute=impute, indicators=("special_requests",), derived=derived)
        Xtr = pipe.fit_transform(dev.iloc[tr], Y[tr])
        FOLDS.append((Xtr, Y[tr], pipe.transform(dev.iloc[va]), va, pipe.categorical_indices_))

def run(args):
    fold, depth, leaf, criterion, missing = args
    Xtr, ytr, Xva, va, cat = FOLDS[fold]
    tree = CARTClassifier(criterion, max_depth=depth, min_samples_leaf=leaf, missing=missing, categorical_features=cat).fit(Xtr, ytr)
    return fold, (depth, leaf), {a: tree.pruned_copy(a).predict_proba(Xva)[:, 1] for a in ALPHAS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impute", default="median")
    ap.add_argument("--derived", default="all", help="'all', 'none', atau daftar dipisah koma")
    ap.add_argument("--criterion", default="gini")
    ap.add_argument("--missing", default="learned")
    ap.add_argument("--depths", default="8,10,12,14,18,none")
    ap.add_argument("--leaves", default="1,3,5,10,20")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    derived = None if a.derived == "all" else ([] if a.derived == "none" else a.derived.split(","))
    prepare(a.impute, derived, a.seed)
    depths = [None if d == "none" else int(d) for d in a.depths.split(",")]
    leaves = [int(v) for v in a.leaves.split(",")]
    tasks = [(f, d, l, a.criterion, a.missing) for d, l in itertools.product(depths, leaves) for f in range(5)]
    oof = {}
    t0 = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        for fold, key, scores in pool.imap_unordered(run, tasks):
            for alpha, s in scores.items():
                oof.setdefault((*key, alpha), np.zeros(len(Y)))[FOLDS[fold][3]] = s
    rows = []
    for (depth, leaf, alpha), s in oof.items():
        thr, best = best_threshold(Y, s)
        rows.append({"max_depth": depth, "min_samples_leaf": leaf, "ccp_alpha": alpha,
                     "macro_f1_at_half": macro_f1(Y, (s >= 0.5).astype(int)),
                     "best_macro_f1": best, "threshold": thr})
    rows.sort(key=lambda r: -r["best_macro_f1"])
    print(f"{len(rows)} kandidat dalam {time.time() - t0:.0f}s ({vars(a)})")
    for r in rows[:12]:
        print(r)
    if a.out:
        Path(a.out).write_text(json.dumps({"args": vars(a), "results": rows}, indent=2))

if __name__ == "__main__":
    main()