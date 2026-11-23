from __future__ import annotations
import numpy as np
from .base import BaseClassifier


def quantile_bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    known = x[~np.isnan(x)]
    if len(known) == 0:
        return np.array([])
    edges = np.unique(np.quantile(known, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return edges


def apply_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.searchsorted(edges, x, side="right").astype(float)
    out[np.isnan(x)] = np.nan
    return out


class _Discretizer:

    def __init__(self, categorical, n_bins, max_unique_as_discrete=12):
        self.categorical = set(categorical)
        self.n_bins = n_bins
        self.max_unique = max_unique_as_discrete

    def fit(self, X):
        self.maps_ = []
        for f in range(X.shape[1]):
            x = X[:, f]
            known = np.unique(x[~np.isnan(x)])
            if f in self.categorical or len(known) <= self.max_unique:
                self.maps_.append(("values", known))
            else:
                self.maps_.append(("bins", quantile_bin_edges(x, self.n_bins)))
        return self

    def transform(self, X):
        out = np.full(X.shape, np.nan)
        for f, (kind, ref) in enumerate(self.maps_):
            x = X[:, f]
            if kind == "values":
                pos = np.searchsorted(ref, x)
                pos_clip = np.clip(pos, 0, max(len(ref) - 1, 0))
                ok = ~np.isnan(x) & (len(ref) > 0)
                if len(ref):
                    ok &= ref[pos_clip] == x
                out[ok, f] = pos_clip[ok]
            else:
                out[:, f] = apply_bins(x, ref)
        return out

    def cardinalities(self):
        return [len(ref) if kind == "values" else len(ref) + 1 for kind, ref in self.maps_]


class NaiveBayes(BaseClassifier):

    def __init__(self, continuous="discretize", n_bins=10, alpha=1.0, var_smoothing=1e-9,
                 categorical_features=None, feature_mask=None, priors=None):
        self.continuous = continuous
        self.n_bins = n_bins
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        self.categorical_features = categorical_features
        self.feature_mask = feature_mask
        self.priors = priors

    def fit(self, X, y, categorical_features=None):
        X = self._check_X(X, allow_nan=True)
        y = self._encode_y(y)

        if categorical_features is not None:
            self.categorical_features = list(categorical_features)
        cat = set(self.categorical_features or [])
        self.n_features_ = X.shape[1]
        K = len(self.classes_)
        counts = np.bincount(y, minlength=K).astype(float)
        self.class_log_prior_ = np.log(counts / counts.sum() if self.priors is None else np.asarray(self.priors, float))

        if self.continuous == "discretize":
            self.discretizer_ = _Discretizer(cat, self.n_bins).fit(X)
            Xd = self.discretizer_.transform(X)
            self.discrete_ = list(range(self.n_features_))
            self.gaussian_ = []
        elif self.continuous == "gaussian":
            self.discretizer_ = None
            Xd = X
            self.discrete_ = sorted(cat)
            self.gaussian_ = [f for f in range(self.n_features_) if f not in cat]
        else:
            raise ValueError(f"unknown continuous: {self.continuous}")

        self.log_tables_ = {}
        for f in self.discrete_:
            x = Xd[:, f]
            known = ~np.isnan(x)
            m = int(np.nanmax(x)) + 1 if known.any() else 1
            if self.discretizer_ is not None:
                m = max(m, self.discretizer_.cardinalities()[f])
            table = np.zeros((K, m))
            np.add.at(table, (y[known], x[known].astype(int)), 1.0)
            table += self.alpha
            self.log_tables_[f] = np.log(table / table.sum(axis=1, keepdims=True))

        if self.gaussian_:
            G = X[:, self.gaussian_]
            eps = self.var_smoothing * np.nanvar(G, axis=0).max()
            self.theta_ = np.zeros((K, len(self.gaussian_)))
            self.var_ = np.zeros((K, len(self.gaussian_)))
            for c in range(K):
                Gc = G[y == c]
                self.theta_[c] = np.nanmean(Gc, axis=0)
                self.var_[c] = np.nanvar(Gc, axis=0) + eps
        return self

    def _used(self, f):
        return self.feature_mask is None or bool(self.feature_mask[f])

    def joint_log_likelihood(self, X):
        self._check_fitted()
        X = self._check_X(X, allow_nan=True)
        Xd = self.discretizer_.transform(X) if self.discretizer_ is not None else X
        jll = np.tile(self.class_log_prior_, (len(X), 1))
        for f, table in self.log_tables_.items():
            if not self._used(f):
                continue
            x = Xd[:, f]
            ok = ~np.isnan(x) & (x >= 0) & (x < table.shape[1])
            jll[ok] += table[:, x[ok].astype(int)].T
        for g, f in enumerate(self.gaussian_):
            if not self._used(f):
                continue
            x = X[:, f]
            ok = ~np.isnan(x)
            mu, var = self.theta_[:, g], self.var_[:, g]
            jll[ok] += -0.5 * (np.log(2 * np.pi * var)[None, :]
                               + (x[ok, None] - mu[None, :]) ** 2 / var[None, :])
        return jll

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)


class TreeAugmentedNaiveBayes(BaseClassifier):

    def __init__(self, n_bins=10, alpha=1.0, categorical_features=None, root=None):
        self.n_bins = n_bins
        self.alpha = alpha
        self.categorical_features = categorical_features
        self.root = root

    @staticmethod
    def conditional_mutual_information(xi, xj, y, mi, mj, K):
        ok = ~np.isnan(xi) & ~np.isnan(xj)
        a, b, c = xi[ok].astype(int), xj[ok].astype(int), y[ok]
        n = len(c)
        if n == 0:
            return 0.0
        joint = np.zeros((K, mi, mj))
        np.add.at(joint, (c, a, b), 1.0)
        p_cab = joint / n
        p_c = p_cab.sum(axis=(1, 2), keepdims=True)
        p_ca = p_cab.sum(axis=2, keepdims=True)
        p_cb = p_cab.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p_cab * p_c / (p_ca * p_cb)
            terms = np.where(p_cab > 0, p_cab * np.log(ratio), 0.0)
        return float(terms.sum())

    @staticmethod
    def maximum_spanning_tree(weights, root):
        d = len(weights)
        parent = np.full(d, -1)
        in_tree = np.zeros(d, dtype=bool)
        in_tree[root] = True
        best = weights[root].copy()
        best_from = np.full(d, root)
        for _ in range(d - 1):
            cand = np.where(in_tree, -np.inf, best)
            v = int(np.argmax(cand))
            in_tree[v] = True
            parent[v] = best_from[v]
            better = (~in_tree) & (weights[v] > best)
            best[better] = weights[v][better]
            best_from[better] = v
        return parent

    def fit(self, X, y, categorical_features=None):
        X = self._check_X(X, allow_nan=True)
        y = self._encode_y(y)

        if categorical_features is not None:
            self.categorical_features = list(categorical_features)

        self.discretizer_ = _Discretizer(set(self.categorical_features or []), self.n_bins).fit(X)
        Xd = self.discretizer_.transform(X)
        self.card_ = self.discretizer_.cardinalities()
        d, K = X.shape[1], len(self.classes_)
        self.n_features_ = d
        counts = np.bincount(y, minlength=K).astype(float)
        self.class_log_prior_ = np.log(counts / counts.sum())

        self.cmi_ = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                v = self.conditional_mutual_information(Xd[:, i], Xd[:, j], y, self.card_[i], self.card_[j], K)
                self.cmi_[i, j] = self.cmi_[j, i] = v
        self.mi_class_ = np.array([self._mutual_information_with_class(Xd[:, i], y, self.card_[i], K) for i in range(d)])
        root = int(np.argmax(self.mi_class_)) if self.root is None else self.root
        self.root_ = root
        self.parent_ = self.maximum_spanning_tree(self.cmi_, root)

        a = self.alpha
        self.single_ = []   # log P(x_i | C)
        self.pair_ = []     # log P(x_i | C, x_parent)
        for i in range(d):
            xi = Xd[:, i]
            ok = ~np.isnan(xi)
            t = np.zeros((K, self.card_[i]))
            np.add.at(t, (y[ok], xi[ok].astype(int)), 1.0)
            t += a
            self.single_.append(np.log(t / t.sum(axis=1, keepdims=True)))
            p = self.parent_[i]
            if p < 0:
                self.pair_.append(None)
                continue
            xp = Xd[:, p]
            ok2 = ok & ~np.isnan(xp)
            t2 = np.zeros((K, self.card_[p], self.card_[i]))
            np.add.at(t2, (y[ok2], xp[ok2].astype(int), xi[ok2].astype(int)), 1.0)
            t2 += a
            self.pair_.append(np.log(t2 / t2.sum(axis=2, keepdims=True)))
        return self

    @staticmethod
    def _mutual_information_with_class(x, y, m, K):
        ok = ~np.isnan(x)
        joint = np.zeros((K, m))
        np.add.at(joint, (y[ok], x[ok].astype(int)), 1.0)
        p = joint / max(joint.sum(), 1)
        pc = p.sum(axis=1, keepdims=True)
        px = p.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p / (pc * px)), 0.0)
        return float(terms.sum())

    def joint_log_likelihood(self, X):
        self._check_fitted()
        Xd = self.discretizer_.transform(self._check_X(X, allow_nan=True))
        jll = np.tile(self.class_log_prior_, (len(Xd), 1))
        for i in range(self.n_features_):
            xi = Xd[:, i]
            ok = ~np.isnan(xi)
            p = self.parent_[i]
            if p >= 0:
                xp = Xd[:, p]
                both = ok & ~np.isnan(xp)
                jll[both] += self.pair_[i][:, xp[both].astype(int), xi[both].astype(int)].T
                ok = ok & np.isnan(xp)
            jll[ok] += self.single_[i][:, xi[ok].astype(int)].T
        return jll

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)
