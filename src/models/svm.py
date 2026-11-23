from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .base import BaseClassifier

_TAU = 1e-12


# source (LRU kernel cache): Chang & Lin (2011), "LIBSVM: A Library for Support Vector Machines", ACM Transactions on Intelligent Systems and Technology 2(3).
class _KernelRows:

    def __init__(self, X, kernel, gamma, degree, coef0, cache_mb):
        self.X = X
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.sq = np.einsum("ij,ij->i", X, X)
        row_bytes = max(len(X) * 4, 1)
        self.capacity = max(int(cache_mb * 1e6 // row_bytes), 2)
        self.cache = OrderedDict()
        self.misses = 0

    def compute(self, A, A_sq=None):
        return kernel_matrix(A, self.X, self.kernel, self.gamma, self.degree, self.coef0,
                             A_sq=A_sq, B_sq=self.sq)

    def row(self, i):
        cached = self.cache.get(i)
        if cached is not None:
            self.cache.move_to_end(i)
            return cached
        self.misses += 1
        values = self.compute(self.X[i:i + 1], self.sq[i:i + 1])[0].astype(np.float32)
        self.cache[i] = values
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return values

    def diagonal(self):
        if self.kernel == "rbf":
            return np.ones(len(self.X))
        if self.kernel == "linear":
            return self.sq.copy()
        return (self.gamma * self.sq + self.coef0) ** self.degree


def kernel_matrix(A, B, kernel="rbf", gamma=1.0, degree=3, coef0=0.0, A_sq=None, B_sq=None):
    dot = A @ B.T
    if kernel == "linear":
        return dot
    if kernel == "poly":
        return (gamma * dot + coef0) ** degree
    if kernel == "rbf":
        if A_sq is None:
            A_sq = np.einsum("ij,ij->i", A, A)
        if B_sq is None:
            B_sq = np.einsum("ij,ij->i", B, B)
        sq_dist = np.maximum(A_sq[:, None] + B_sq[None, :] - 2.0 * dot, 0.0)
        return np.exp(-gamma * sq_dist)
    raise ValueError(f"kernel tidak dikenal: {kernel}")


# source: IF3170 lecture slides, Materi 12 (SVM).
# source: Platt (1998), "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines", Microsoft Research Technical Report MSR-TR-98-14.
class SVC(BaseClassifier):

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 tol=1e-3, max_iter=-1, class_weight=None, cache_mb=1000, verbose=False):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.cache_mb = cache_mb
        self.verbose = verbose

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    # source (working set selection WSS2): Fan, Chen & Lin (2005), "Working Set Selection Using Second Order Information for Training Support Vector Machines", JMLR 6.
    # source (two-variable update and bias): Chang & Lin (2011), "LIBSVM: A Library for Support Vector Machines", ACM Transactions on Intelligent Systems and Technology 2(3).
    def fit(self, X, y):
        X = self._check_X(X)
        codes = self._encode_y(y)
        if len(self.classes_) != 2:
            raise ValueError("SVC ini biner; gunakan skema one-vs-rest untuk multikelas")
        yy = np.where(codes == 1, 1.0, -1.0)
        n = len(yy)
        self.gamma_ = self._resolve_gamma(X)
        cw = self._class_weight_vector(self.class_weight, 2)
        Cvec = np.where(yy > 0, self.C * cw[1], self.C * cw[0])
        rows = _KernelRows(X, self.kernel, self.gamma_, self.degree, self.coef0, self.cache_mb)
        QD = rows.diagonal()

        alpha = np.zeros(n)
        G = -np.ones(n)
        it = 0
        max_iter = np.inf if self.max_iter is None or self.max_iter < 0 else self.max_iter
        while it < max_iter:
            minus_yG = -yy * G
            up = ((yy > 0) & (alpha < Cvec)) | ((yy < 0) & (alpha > 0))
            low = ((yy > 0) & (alpha > 0)) | ((yy < 0) & (alpha < Cvec))
            if not up.any() or not low.any():
                break
            cand = np.where(up, minus_yG, -np.inf)
            i = int(np.argmax(cand))
            g_max = cand[i]
            g_min_low = np.where(low, minus_yG, np.inf)
            if g_max - g_min_low.min() < self.tol:
                break
            Ki = rows.row(i).astype(float)
            grad_diff = g_max - minus_yG
            quad = QD[i] + QD - 2.0 * Ki
            quad = np.where(quad > 0, quad, _TAU)
            obj = np.where(low & (grad_diff > 0), -(grad_diff**2) / quad, np.inf)
            j = int(np.argmin(obj))
            if not np.isfinite(obj[j]):
                break
            Kj = rows.row(j).astype(float)

            yi, yj = yy[i], yy[j]
            Ci, Cj = Cvec[i], Cvec[j]
            ai_old, aj_old = alpha[i], alpha[j]
            Qij = yi * yj * Ki[j]
            if yi != yj:
                q = QD[i] + QD[j] + 2.0 * Qij
                q = q if q > 0 else _TAU
                delta = (-G[i] - G[j]) / q
                diff = alpha[i] - alpha[j]
                alpha[i] += delta
                alpha[j] += delta
                if diff > 0:
                    if alpha[j] < 0:
                        alpha[j], alpha[i] = 0.0, diff
                else:
                    if alpha[i] < 0:
                        alpha[i], alpha[j] = 0.0, -diff
                if diff > Ci - Cj:
                    if alpha[i] > Ci:
                        alpha[i], alpha[j] = Ci, Ci - diff
                else:
                    if alpha[j] > Cj:
                        alpha[j], alpha[i] = Cj, Cj + diff
            else:
                q = QD[i] + QD[j] - 2.0 * Qij
                q = q if q > 0 else _TAU
                delta = (G[i] - G[j]) / q
                total = alpha[i] + alpha[j]
                alpha[i] -= delta
                alpha[j] += delta
                if total > Ci:
                    if alpha[i] > Ci:
                        alpha[i], alpha[j] = Ci, total - Ci
                else:
                    if alpha[j] < 0:
                        alpha[j], alpha[i] = 0.0, total
                if total > Cj:
                    if alpha[j] > Cj:
                        alpha[j], alpha[i] = Cj, total - Cj
                else:
                    if alpha[i] < 0:
                        alpha[i], alpha[j] = 0.0, total
            d_i, d_j = alpha[i] - ai_old, alpha[j] - aj_old
            G += yy * (yi * d_i * Ki + yj * d_j * Kj)
            it += 1
            if self.verbose and it % 5000 == 0:
                print(f"  SMO iter {it}: gap={g_max - g_min_low.min():.5f}, cache misses={rows.misses}")

        self.n_iter_ = it
        self.cache_misses_ = rows.misses
        self.rho_ = self._compute_rho(yy, G, alpha, Cvec)
        sv = alpha > 0
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (alpha * yy)[sv]
        self.alpha_ = alpha
        self.intercept_ = -self.rho_
        self.n_support_ = np.array([np.sum(sv & (yy < 0)), np.sum(sv & (yy > 0))])
        margin = G + 1.0 - yy * self.rho_
        self.slack_ = np.maximum(0.0, 1.0 - margin)
        self.slack_sum_ = float(self.slack_.sum())
        self.dual_objective_ = float(alpha.sum() - 0.5 * np.dot(alpha, G + 1.0))
        if self.kernel == "linear":
            self.coef_ = self.dual_coef_ @ self.support_vectors_
        return self

    @staticmethod
    def _compute_rho(yy, G, alpha, Cvec):
        yG = yy * G
        upper = alpha >= Cvec
        lower = alpha <= 0
        free = ~upper & ~lower
        if free.any():
            return float(yG[free].mean())
        ub_mask = (upper & (yy < 0)) | (lower & (yy > 0))
        lb_mask = (upper & (yy > 0)) | (lower & (yy < 0))
        ub = yG[ub_mask].min() if ub_mask.any() else np.inf
        lb = yG[lb_mask].max() if lb_mask.any() else -np.inf
        return float((ub + lb) / 2)

    def decision_function(self, X, batch_size=2048):
        self._check_fitted()
        X = self._check_X(X)
        if self.kernel == "linear":
            return X @ self.coef_ - self.rho_
        out = np.empty(len(X))
        sv_sq = np.einsum("ij,ij->i", self.support_vectors_, self.support_vectors_)
        for start in range(0, len(X), batch_size):
            block = X[start:start + batch_size]
            K = kernel_matrix(block, self.support_vectors_, self.kernel, self.gamma_,
                              self.degree, self.coef0, B_sq=sv_sq)
            out[start:start + batch_size] = K @ self.dual_coef_ - self.rho_
        return out

    def predict_proba(self, X):
        f = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(f, -50, 50)))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=None):
        f = self.decision_function(X)
        cut = 0.0 if threshold is None else np.log(threshold / (1 - threshold))
        return self.classes_[(f >= cut).astype(int)]
