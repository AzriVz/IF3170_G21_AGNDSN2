from __future__ import annotations

import numpy as np

from .base import BaseClassifier


def smallest_k_indices(D: np.ndarray, k: int) -> np.ndarray:
    n_rows = D.shape[0]
    part = np.argpartition(D, k - 1, axis=1)[:, :k]
    kth = np.take_along_axis(D, part, axis=1).max(axis=1)
    less = D < kth[:, None]
    need = k - less.sum(axis=1)
    equal = D == kth[:, None]
    take_equal = equal & (np.cumsum(equal, axis=1) <= need[:, None])
    chosen = np.nonzero(less | take_equal)[1].reshape(n_rows, k)
    chosen_d = np.take_along_axis(D, chosen, axis=1)
    order = np.lexsort((chosen, chosen_d), axis=1)
    return np.take_along_axis(chosen, order, axis=1)


def elementwise_sq_distances(Q: np.ndarray, X: np.ndarray, weights=None) -> np.ndarray:
    out = np.zeros((len(Q), len(X)))
    buf = np.empty_like(out)
    w = np.ones(Q.shape[1]) if weights is None else weights
    for f in range(Q.shape[1]):
        np.subtract(Q[:, f][:, None], X[:, f][None, :], out=buf)
        np.square(buf, out=buf)
        if w[f] != 1.0:
            buf *= w[f]
        out += buf
    return out


# source: IF3170 lecture slides, Materi 10b (kNN).
# source (distance-weighted vote): Dudani (1976), "The Distance-Weighted k-Nearest-Neighbor Rule", IEEE Trans. Systems, Man, and Cybernetics SMC-6(4).
# source (feature weights): Wettschereck, Aha & Mohri (1997), "A Review and Empirical Evaluation of Feature Weighting Methods for a Class of Lazy Learning Algorithms", Artificial Intelligence Review 11.
# source (missing-value distance 'max'): Wilson & Martinez (1997), "Improved Heterogeneous Distance Functions", Journal of Artificial Intelligence Research 6.
class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors=5, metric="euclidean", p=2, weights="uniform",
                 feature_weights=None, categorical_features=None, missing="ignore",
                 missing_penalty=1.0, batch_size=1024, distance_power=1.0):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.feature_weights = feature_weights
        self.categorical_features = categorical_features
        self.missing = missing
        self.missing_penalty = missing_penalty
        self.batch_size = batch_size
        self.distance_power = distance_power

    def fit(self, X, y, categorical_features=None):
        X = self._check_X(X, allow_nan=True)
        self._y = self._encode_y(y)
        if len(X) != len(self._y):
            raise ValueError("Jumlah baris X dan y berbeda")
        if categorical_features is not None:
            self.categorical_features = list(categorical_features)
        d = X.shape[1]
        self._is_cat = np.zeros(d, dtype=bool)
        if self.categorical_features:
            self._is_cat[list(self.categorical_features)] = True
        fw = np.ones(d) if self.feature_weights is None else np.asarray(self.feature_weights, float)
        if fw.shape != (d,):
            raise ValueError("feature_weights harus berpanjang sama dengan jumlah fitur")
        self._fw = fw
        self._order = {"euclidean": 2.0, "manhattan": 1.0, "minkowski": float(self.p)}[self.metric]
        self._X = X
        self._col_nan = np.isnan(X).T.copy()
        self._has_nan = bool(self._col_nan.any())
        if self._order == 2.0 and not self._is_cat.any():
            self._Xs = X * np.sqrt(fw)
            self._sq_norm = np.einsum("ij,ij->i", self._Xs, self._Xs)
        return self

    def _block_distances(self, Q: np.ndarray) -> np.ndarray:
        use_fast = (self._order == 2.0 and not self._is_cat.any()
                    and not self._has_nan and not np.isnan(Q).any())
        if use_fast:
            Qs = Q * np.sqrt(self._fw)
            sq = np.einsum("ij,ij->i", Qs, Qs)[:, None] + self._sq_norm[None, :] - 2.0 * (Qs @ self._Xs.T)
            np.maximum(sq, 0.0, out=sq)
            return np.sqrt(sq)
        if self._order == 2.0 and not self._is_cat.any() and self.missing == "ignore":
            return self._masked_euclidean(Q)
        n, d = len(self._X), self._X.shape[1]
        total = np.zeros((len(Q), n))
        buf = np.empty((len(Q), n))
        used = None
        if self.missing == "ignore" and (self._has_nan or np.isnan(Q).any()):
            used = np.zeros((len(Q), n))
        for f in range(d):
            w = self._fw[f]
            if w == 0:
                continue
            q = Q[:, f][:, None]
            x = self._X[:, f][None, :]
            if self._is_cat[f]:
                buf[...] = q != x
            else:
                np.subtract(q, x, out=buf)
                np.abs(buf, out=buf)
                if self._order == 2.0:
                    np.square(buf, out=buf)
                elif self._order != 1.0:
                    np.power(buf, self._order, out=buf)
            q_nan, x_nan = np.isnan(Q[:, f]), self._col_nan[f]
            if q_nan.any() or x_nan.any():
                nan = q_nan[:, None] | x_nan[None, :]
                if self.missing == "ignore":
                    buf[nan] = 0.0
                    used += w * ~nan
                else:
                    buf[nan] = self.missing_penalty ** self._order
            elif used is not None:
                used += w
            buf *= w
            total += buf
        if used is not None:
            full = self._fw[self._fw != 0].sum()
            total *= full / np.maximum(used, 1e-12)
            total[used <= 0] = np.inf
        if self._order != 1.0:
            total **= 1.0 / self._order
        return total

    # source: Dixon (1979), "Pattern Recognition with Partly Missing Data", IEEE Trans. Systems, Man, and Cybernetics 9(10).
    def _masked_euclidean(self, Q: np.ndarray) -> np.ndarray:
        w = self._fw
        mq = (~np.isnan(Q)).astype(float)
        mx = (~self._col_nan).T.astype(float)
        q0 = np.nan_to_num(Q)
        x0 = np.nan_to_num(self._X)
        sq = (w * q0 * q0) @ mx.T + mq @ (w * x0 * x0).T - 2.0 * (w * q0) @ x0.T
        np.maximum(sq, 0.0, out=sq)
        used = mq @ (w * mx).T
        sq *= w[w != 0].sum() / np.maximum(used, 1e-12)
        sq[used <= 0] = np.inf
        return np.sqrt(sq)

    def kneighbors(self, X, n_neighbors=None, exclude_self=False):
        self._check_fitted()
        X = self._check_X(X, allow_nan=True)
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        k_eff = k + 1 if exclude_self else k
        if k_eff > len(self._X):
            raise ValueError("k melebihi jumlah data latih")
        dist_out = np.empty((len(X), k))
        idx_out = np.empty((len(X), k), dtype=np.int64)
        for start in range(0, len(X), self.batch_size):
            Q = X[start:start + self.batch_size]
            D = self._block_distances(Q)
            if exclude_self:
                rows = np.arange(len(Q))
                D[rows, start + rows] = np.inf
            nearest = smallest_k_indices(D, k)
            idx_out[start:start + len(Q)] = nearest
            dist_out[start:start + len(Q)] = np.take_along_axis(D, nearest, axis=1)
        return dist_out, idx_out

    def _vote(self, dist: np.ndarray, idx: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        labels = self._y[idx]
        if self.weights == "uniform":
            w = np.ones_like(dist)
        elif self.weights == "distance":
            with np.errstate(divide="ignore"):
                w = 1.0 / dist ** getattr(self, "distance_power", 1.0)
            zero = dist == 0
            rows = zero.any(axis=1)
            w[rows] = zero[rows].astype(float)
        else:
            raise ValueError(f"weights tidak dikenal: {self.weights}")
        proba = np.zeros((len(dist), n_classes))
        for c in range(n_classes):
            proba[:, c] = (w * (labels == c)).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        return self._vote(dist, idx)

    def predict_proba_multi_k(self, X, ks):
        dist, idx = self.kneighbors(X, n_neighbors=max(ks))
        return {k: self._vote(dist[:, :k], idx[:, :k]) for k in ks}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_Xs", None)
        state.pop("_sq_norm", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "_X") and self._order == 2.0 and not self._is_cat.any():
            self._Xs = self._X * np.sqrt(self._fw)
            self._sq_norm = np.einsum("ij,ij->i", self._Xs, self._Xs)
