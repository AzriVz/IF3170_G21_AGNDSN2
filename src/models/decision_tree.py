from __future__ import annotations

import copy
import numpy as np
from .base import BaseClassifier

_EPS = 1e-12
def _impurity(counts: np.ndarray, criterion: str) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=total > 0)
    if criterion == "gini":
        return 1.0 - np.sum(p * p, axis=-1)
    if criterion == "entropy":
        logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
        return -np.sum(p * logp, axis=-1)
    raise ValueError(f"criterion tidak dikenal: {criterion}")


class _Node:
    __slots__ = ("node_id", "depth", "value", "weight", "impurity", "feature", "threshold", "left_categories", "known_categories", "nan_left", "left_fraction", "left", "right", "decrease", "n_samples")

    def __init__(self, node_id, depth, value, weight, impurity, n_samples):
        self.node_id = node_id
        self.depth = depth
        self.value = value            
        self.weight = weight         
        self.impurity = impurity
        self.n_samples = n_samples
        self.feature = None
        self.threshold = None        
        self.left_categories = None  
        self.known_categories = None
        self.nan_left = True
        self.left_fraction = 0.5 
        self.left = None
        self.right = None
        self.decrease = 0.0

    @property
    def is_leaf(self):
        return self.left is None

    def make_leaf(self):
        self.left = self.right = None
        self.feature = self.threshold = self.left_categories = self.known_categories = None


class CARTClassifier(BaseClassifier):
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0, missing="learned",
                 ccp_alpha=0.0, class_weight=None, categorical_features=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.missing = missing
        self.ccp_alpha = ccp_alpha
        self.class_weight = class_weight
        self.categorical_features = categorical_features

    def fit(self, X, y, sample_weight=None, categorical_features=None):
        X = self._check_X(X, allow_nan=True)
        y = self._encode_y(y)
        if categorical_features is not None:
            self.categorical_features = list(categorical_features)
        if self.missing not in ("fractional", "majority", "learned"):
            raise ValueError(f"missing tidak dikenal: {self.missing}")
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(self.classes_)
        self._is_cat = np.zeros(self.n_features_, dtype=bool)
        if self.categorical_features:
            self._is_cat[list(self.categorical_features)] = True
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float).copy()
        w *= self._class_weight_vector(self.class_weight, self.n_classes_)[y]
        self._X, self._y = X, y
        self._root_weight = w.sum()
        self._next_id = 0
        self.root_ = self._grow(np.arange(len(y)), w, depth=0)
        del self._X, self._y
        if self.ccp_alpha > 0:
            self.prune_cost_complexity(self.ccp_alpha)
        return self

    def _new_node(self, idx, w, depth):
        value = np.bincount(self._y[idx], weights=w, minlength=self.n_classes_).astype(float)
        node = _Node(self._next_id, depth, value, float(w.sum()),
                     float(_impurity(value, self.criterion)), len(idx))
        self._next_id += 1
        return node

    def _grow(self, idx, w, depth):
        node = self._new_node(idx, w, depth)
        if (node.impurity <= _EPS
                or (self.max_depth is not None and depth >= self.max_depth)
                or node.weight < self.min_samples_split
                or node.weight < 2 * self.min_samples_leaf):
            return node
        split = self._best_split(idx, w, node)
        if split is None:
            return node
        decrease = node.weight / self._root_weight * split["score"]
        if decrease + _EPS < self.min_impurity_decrease:
            return node
        node.feature = split["feature"]
        node.threshold = split.get("threshold")
        node.left_categories = split.get("left_categories")
        node.known_categories = split.get("known_categories")
        node.nan_left = split["nan_left"]
        node.left_fraction = split["left_fraction"]
        node.decrease = decrease
        (li, lw), (ri, rw) = self._partition(self._X[idx, node.feature], idx, w, node)
        node.left = self._grow(li, lw, depth + 1)
        node.right = self._grow(ri, rw, depth + 1)
        return node

    def _goes_left(self, x, node):
        if node.threshold is not None:
            known = ~np.isnan(x)
            left = known & (x <= node.threshold)
        else:
            known = np.isin(x, node.known_categories)
            left = np.isin(x, node.left_categories)
        return left, known

    def _partition(self, x, idx, w, node):
        left, known = self._goes_left(x, node)
        right = known & ~left
        missing = ~known
        if self.missing == "fractional":
            lw = np.concatenate([w[left], w[missing] * node.left_fraction])
            rw = np.concatenate([w[right], w[missing] * (1 - node.left_fraction)])
            li = np.concatenate([idx[left], idx[missing]])
            ri = np.concatenate([idx[right], idx[missing]])
            keep_l, keep_r = lw > _EPS, rw > _EPS
            return (li[keep_l], lw[keep_l]), (ri[keep_r], rw[keep_r])
        nan_left = bool(node.nan_left)
        left_all = left | (missing & nan_left)
        right_all = right | (missing & (not nan_left))
        return (idx[left_all], w[left_all]), (idx[right_all], w[right_all])

    def _best_split(self, idx, w, node):
        y = self._y[idx]
        best = None
        for f in range(self.n_features_):
            x = self._X[idx, f]
            known = ~np.isnan(x)
            if known.sum() < 2:
                continue
            missing_counts = np.bincount(y[~known], weights=w[~known], minlength=self.n_classes_)
            if self._is_cat[f]:
                cand = self._categorical_candidates(x[known], y[known], w[known])
            else:
                cand = self._numeric_candidates(x[known], y[known], w[known])
            if cand is None:
                continue
            left_counts, extra = cand
            result = self._score_candidates(left_counts, missing_counts, node)
            if result is None:
                continue
            score, pos, nan_left, left_fraction = result
            if best is None or score > best["score"] + _EPS:
                best = {"feature": f, "score": score, "nan_left": nan_left,
                        "left_fraction": left_fraction}
                if self._is_cat[f]:
                    order, present = extra
                    best["left_categories"] = order[:pos + 1].astype(float)
                    best["known_categories"] = present.astype(float)
                else:
                    best["threshold"] = extra[pos]
        return best

    def _numeric_candidates(self, x, y, w):
        order = np.argsort(x, kind="mergesort")
        xs, ys, ws = x[order], y[order], w[order]
        onehot = np.zeros((len(xs), self.n_classes_))
        onehot[np.arange(len(xs)), ys] = ws
        cum = np.cumsum(onehot, axis=0)
        boundary = np.flatnonzero(xs[:-1] < xs[1:])
        if len(boundary) == 0:
            return None
        thresholds = (xs[boundary] + xs[boundary + 1]) / 2.0
        thresholds = np.where(thresholds == xs[boundary + 1], xs[boundary], thresholds)
        return cum[boundary], thresholds

    def _categorical_candidates(self, x, y, w):
        codes = x.astype(np.int64)
        present = np.unique(codes)
        if len(present) < 2:
            return None
        n_cat = int(present.max()) + 1
        flat = np.bincount(codes * self.n_classes_ + y, weights=w, minlength=n_cat * self.n_classes_)
        counts = flat.reshape(n_cat, self.n_classes_)[present]
        share = counts[:, -1] / counts.sum(axis=1)
        order = np.argsort(share, kind="mergesort")
        cum = np.cumsum(counts[order], axis=0)[:-1]
        return cum, (present[order], present)

    def _score_candidates(self, left, missing, node):
        total_known = node.value - missing
        right = total_known[None, :] - left
        wl, wr = left.sum(axis=1), right.sum(axis=1)
        w_known = wl + wr
        w_missing = missing.sum()
        crit = self.criterion
        msl = self.min_samples_leaf
        if self.missing == "fractional":
            frac_l = wl / np.maximum(w_known, _EPS)
            eff_l = wl + w_missing * frac_l
            eff_r = wr + w_missing * (1 - frac_l)
            valid = (eff_l >= msl - _EPS) & (eff_r >= msl - _EPS) & (wl > 0) & (wr > 0)
            if not valid.any():
                return None
            child = (wl * _impurity(left, crit) + wr * _impurity(right, crit)) / w_known
            score = (w_known / node.weight) * (_impurity(total_known, crit) - child)
            score = np.where(valid, score, -np.inf)
            pos = int(np.argmax(score))
            return float(score[pos]), pos, bool(frac_l[pos] >= 0.5), float(frac_l[pos])
        parent = node.impurity
        options = []
        for nan_left in (True, False):
            if self.missing == "majority":
                to_left = wl >= wr
                lc = left + np.where(to_left[:, None], missing[None, :], 0)
                rc = right + np.where(to_left[:, None], 0, missing[None, :])
            elif nan_left:
                lc, rc = left + missing[None, :], right
            else:
                lc, rc = left, right + missing[None, :]
            wl2, wr2 = lc.sum(axis=1), rc.sum(axis=1)
            valid = (wl2 >= msl - _EPS) & (wr2 >= msl - _EPS) & (wl > 0) & (wr > 0)
            child = (wl2 * _impurity(lc, crit) + wr2 * _impurity(rc, crit)) / node.weight
            score = np.where(valid, parent - child, -np.inf)
            nan_dir = to_left if self.missing == "majority" else np.full(len(wl), nan_left)
            options.append((score, nan_dir))
            if self.missing == "majority" or w_missing == 0:
                break
        best_score, best_pos, best_dir = -np.inf, None, True
        for score, nan_dir in options:
            pos = int(np.argmax(score))
            if score[pos] > best_score + _EPS:
                best_score, best_pos, best_dir = score[pos], pos, bool(nan_dir[pos])
        if best_pos is None or not np.isfinite(best_score):
            return None
        if w_missing == 0 and self.missing == "learned":
            best_dir = bool(wl[best_pos] >= wr[best_pos])
        frac = float(wl[best_pos] / max(w_known[best_pos], _EPS))
        return float(best_score), best_pos, best_dir, frac

    def _route(self, X):
        stack = [(self.root_, np.arange(len(X)), np.ones(len(X)))]
        while stack:
            node, idx, w = stack.pop()
            yield node, idx, w
            if node.is_leaf or len(idx) == 0:
                continue
            (li, lw), (ri, rw) = self._partition(X[idx, node.feature], idx, w, node)
            stack.append((node.right, ri, rw))
            stack.append((node.left, li, lw))

    def predict_proba(self, X):
        self._check_fitted()
        X = self._check_X(X, allow_nan=True)
        proba = np.zeros((len(X), self.n_classes_))
        for node, idx, w in self._route(X):
            if node.is_leaf and len(idx):
                dist = node.value / max(node.value.sum(), _EPS)
                proba[idx] += w[:, None] * dist[None, :]
        return proba / proba.sum(axis=1, keepdims=True)

    def apply(self, X):
        X = self._check_X(X, allow_nan=True)
        leaf = np.full(len(X), -1)
        for node, idx, _ in self._route(X):
            if node.is_leaf:
                leaf[idx] = node.node_id
        return leaf

    def nodes(self):
        out, stack = [], [self.root_]
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.is_leaf:
                stack.extend([node.right, node.left])
        return out

    def get_depth(self):
        return max(n.depth for n in self.nodes())

    def get_n_leaves(self):
        return sum(n.is_leaf for n in self.nodes())

    @property
    def feature_importances_(self):
        imp = np.zeros(self.n_features_)
        for node in self.nodes():
            if not node.is_leaf:
                imp[node.feature] += node.decrease
        return imp / imp.sum() if imp.sum() > 0 else imp

    def _subtree_stats(self, node):
        if node.is_leaf:
            return node.weight / self._root_weight * node.impurity, 1
        rl, nl = self._subtree_stats(node.left)
        rr, nr = self._subtree_stats(node.right)
        return rl + rr, nl + nr

    def _weakest_links(self):
        best_alpha, best_nodes = np.inf, []
        stats = {}

        def visit(node):
            if node.is_leaf:
                r = node.weight / self._root_weight * node.impurity
                stats[node.node_id] = (r, 1)
                return r, 1
            rl, nl = visit(node.left)
            rr, nr = visit(node.right)
            r_sub, leaves = rl + rr, nl + nr
            stats[node.node_id] = (r_sub, leaves)
            return r_sub, leaves

        visit(self.root_)
        for node in self.nodes():
            if node.is_leaf:
                continue
            r_node = node.weight / self._root_weight * node.impurity
            r_sub, leaves = stats[node.node_id]
            alpha = (r_node - r_sub) / (leaves - 1)
            if alpha < best_alpha - 1e-15:
                best_alpha, best_nodes = alpha, [node]
            elif abs(alpha - best_alpha) <= 1e-15:
                best_nodes.append(node)
        return best_alpha, best_nodes

    def prune_cost_complexity(self, ccp_alpha):
        while not self.root_.is_leaf:
            alpha, nodes = self._weakest_links()
            if alpha > ccp_alpha:
                break
            for node in nodes:
                node.make_leaf()
        self.ccp_alpha = ccp_alpha
        return self

    def cost_complexity_pruning_path(self):
        tree = copy.deepcopy(self)
        alphas, impurities, leaves = [0.0], [], []
        r, n = tree._subtree_stats(tree.root_)
        impurities.append(r)
        leaves.append(n)
        while not tree.root_.is_leaf:
            alpha, nodes = tree._weakest_links()
            for node in nodes:
                node.make_leaf()
            r, n = tree._subtree_stats(tree.root_)
            alphas.append(max(alpha, 0.0))
            impurities.append(r)
            leaves.append(n)
        return {"ccp_alphas": np.array(alphas), "impurities": np.array(impurities),
                "n_leaves": np.array(leaves)}

    def pruned_copy(self, ccp_alpha):
        tree = copy.deepcopy(self)
        return tree.prune_cost_complexity(ccp_alpha)

    def prune_reduced_error(self, X_val, y_val):
        X_val = self._check_X(X_val, allow_nan=True)
        y_val = np.searchsorted(self.classes_, np.asarray(y_val))
        reached = {}
        for node, idx, w in self._route(X_val):
            reached[node.node_id] = np.bincount(y_val[idx], weights=w, minlength=self.n_classes_)

        def visit(node):
            counts = reached.get(node.node_id, np.zeros(self.n_classes_))
            leaf_error = counts.sum() - counts[int(np.argmax(node.value))]
            if node.is_leaf:
                return leaf_error
            subtree_error = visit(node.left) + visit(node.right)
            if leaf_error <= subtree_error + _EPS:
                node.make_leaf()
                return leaf_error
            return subtree_error

        visit(self.root_)
        return self