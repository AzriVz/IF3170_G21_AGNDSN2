from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import development_split, load_competition_data  # noqa: E402
from src.metrics import best_threshold, macro_f1  # noqa: E402
from src.models import KNNClassifier  # noqa: E402
from src.preprocessing import TARGET, make_pipeline  # noqa: E402
from src.validation import StratifiedKFold  # noqa: E402

FOLDS = []
NAMES = []
Y = None

START_WEIGHTS = {
    "days_before_arrival": 3, "arrival_ord": 3, "nightly_rate_idr": 2, "special_requests": 2,
    "booking_channel=Online Travel Agent": 1, "booking_channel=Offline Agent": 1,
    "booking_channel=Corporate": 1, "total_nights": 1, "num_adults": 1, "needs_parking": 1,
    "is_returning_guest": 1,
}


def prepare(impute, indicators, seed, n_splits=5, knn_impute=()):
    global FOLDS, NAMES, Y
    train, _ = load_competition_data()
    dev, _ = development_split(train)
    Y = dev[TARGET].to_numpy()
    FOLDS = []
    for tr, va in StratifiedKFold(n_splits, True, seed).split(Y):
        pipe = make_pipeline("distance", impute=impute, indicators=indicators, knn_impute=knn_impute)
        Xtr = pipe.fit_transform(dev.iloc[tr], Y[tr])
        Xva = pipe.transform(dev.iloc[va])
        FOLDS.append((Xtr, Y[tr], Xva, va))
        NAMES = pipe.feature_names_


def _fold_scores(args):
    fold, weights, metric, ks, vote, missing, power = args
    Xtr, ytr, Xva, va = FOLDS[fold]
    active = weights > 0
    model = KNNClassifier(max(ks), metric=metric, weights=vote, missing=missing,
                          feature_weights=weights[active], distance_power=power)
    model.fit(Xtr[:, active], ytr)
    probs = model.predict_proba_multi_k(Xva[:, active], ks)
    return fold, {k: p[:, 1] for k, p in probs.items()}


class Evaluator:
    def __init__(self, pool, metric, ks, vote, missing, power=1.0):
        self.pool, self.metric, self.ks, self.vote, self.missing = pool, metric, ks, vote, missing
        self.power = power
        self.calls = 0

    def oof(self, weights):
        tasks = [(f, weights, self.metric, self.ks, self.vote, self.missing, self.power)
                 for f in range(len(FOLDS))]
        out = {k: np.zeros(len(Y)) for k in self.ks}
        for fold, scores in self.pool.imap_unordered(_fold_scores, tasks):
            va = FOLDS[fold][3]
            for k, s in scores.items():
                out[k][va] = s
        self.calls += 1
        return out

    def score(self, weights):
        best = (-1.0, None, None, None)
        for k, s in self.oof(weights).items():
            thr, f1 = best_threshold(Y, s)
            if f1 > best[0]:
                best = (f1, k, thr, macro_f1(Y, (s >= 0.5).astype(int)))
        return best


def coordinate_ascent(ev, weights, grid, rounds, min_gain, log):
    current = ev.score(weights)
    log(f"start: F1={current[0]:.4f} k={current[1]} thr={current[2]}")
    for r in range(rounds):
        improved = False
        for f in range(len(NAMES)):
            best_value, best_score = weights[f], current
            for value in grid:
                if value == weights[f]:
                    continue
                trial = weights.copy()
                trial[f] = value
                if not (trial > 0).any():
                    continue
                score = ev.score(trial)
                if score[0] > best_score[0] + min_gain:
                    best_value, best_score = value, score
            if best_value != weights[f]:
                log(f"  round {r + 1}: {NAMES[f]} {weights[f]} -> {best_value}  "
                    f"F1 {current[0]:.4f} -> {best_score[0]:.4f} (k={best_score[1]})")
                weights[f] = best_value
                current = best_score
                improved = True
        log(f"end round {r + 1}: F1={current[0]:.4f} k={current[1]} thr={current[2]} evals={ev.calls}")
        if not improved:
            break
    return weights, current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impute", default="median")
    ap.add_argument("--indicators", default="special_requests")
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--vote", default="distance")
    ap.add_argument("--missing", default="ignore")
    ap.add_argument("--ks", default="5,7,10,13,16,20,25,30,40")
    ap.add_argument("--grid", default="0,0.5,1,1.5,2,3,4")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default=None, help="JSON berisi bobot awal")
    ap.add_argument("--power", type=float, default=1.0, help="eksponen bobot suara 1/d^p")
    ap.add_argument("--knn-impute", default="", help="kolom yang diimputasi KNN-regresi, dipisah koma")
    ap.add_argument("--start-config", action="store_true", help="mulai dari KNN_WEIGHTS di src/config.py")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    indicators = tuple(c for c in args.indicators.split(",") if c)
    knn_impute = tuple(c for c in args.knn_impute.split(",") if c)
    prepare(args.impute, indicators, args.seed, knn_impute=knn_impute)
    ks = [int(k) for k in args.ks.split(",")]
    grid = [float(g) for g in args.grid.split(",")]
    if args.start:
        start = json.loads(Path(args.start).read_text())["weights"]
    elif args.start_config:
        from src.config import KNN_WEIGHTS
        start = KNN_WEIGHTS
    else:
        start = START_WEIGHTS
    weights = np.array([float(start.get(n, 0.0)) for n in NAMES])

    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"features={len(NAMES)} impute={args.impute} knn_impute={knn_impute} metric={args.metric} "
        f"vote={args.vote} power={args.power} seed={args.seed}")
    t0 = time.time()
    with get_context("fork").Pool(len(FOLDS)) as pool:
        ev = Evaluator(pool, args.metric, ks, args.vote, args.missing, args.power)
        if args.evaluate_only:
            result = ev.score(weights)
            log(f"F1={result[0]:.4f} k={result[1]} thr={result[2]} F1@0.5={result[3]:.4f}")
        else:
            weights, result = coordinate_ascent(ev, weights, grid, args.rounds, args.min_gain, log)
    log(f"done in {time.time() - t0:.0f}s")
    final = {n: float(w) for n, w in zip(NAMES, weights) if w > 0}
    log("weights: " + json.dumps(final))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "args": vars(args), "weights": final, "macro_f1": result[0], "k": result[1],
            "threshold": result[2], "macro_f1_at_half": result[3], "log": lines,
        }, indent=2))


if __name__ == "__main__":
    main()
