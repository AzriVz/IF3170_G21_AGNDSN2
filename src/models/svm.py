from __future__ import annotations

<<<<<<< HEAD
from collections import OrderedDict
=======
import warnings
>>>>>>> 60a0a31 (svm)

import numpy as np

from .base import BaseClassifier

<<<<<<< HEAD
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
=======
# source: IF3170 lecture slides, Materi 12 (Support Vector Machine).
# source: Platt (1998), "Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines", Microsoft Research TR-98-14.
# source: Platt (1998), "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods", Advances in Large Margin Classifiers.
# source: Fan, Chen & Lin (2008), "Working Set Selection Using Second Order Information for Training Support Vector Machines", JMLR 9.

_KERNELS = ("linear", "poly", "rbf")

# rows x support-vectors block budget for decision_function (~32 MB float64)
_DECISION_BLOCK = 4_000_000

# max j candidates tried per SMO step before skipping i to the next pass
_MAX_PAIR_TRIES = 10


def resolve_gamma(gamma, X: np.ndarray) -> float:
    """gamma='scale'/'auto' per sklearn: 1/(n_features * var(X)) and 1/n_features."""
    if isinstance(gamma, str):
        if gamma == "auto":
            return 1.0 / X.shape[1]
        if gamma == "scale":
            var = X.var()
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0 / X.shape[1]
        raise ValueError(f"gamma tidak dikenal: {gamma!r}")
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma harus positif")
    return gamma


def kernel_matrix(A: np.ndarray, B: np.ndarray, kernel: str, gamma: float,
                  degree: int, coef0: float) -> np.ndarray:
    """K(a, b) for every row a of A against every row b of B; blocks only, never n x n."""
    if kernel == "linear":
        return A @ B.T
    G = A @ B.T
    if kernel == "poly":
        return (gamma * G + coef0) ** degree
    a2 = np.einsum("ij,ij->i", A, A)
    b2 = np.einsum("ij,ij->i", B, B)
    d2 = a2[:, None] + b2[None, :] - 2.0 * G
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _fit_platt(f: np.ndarray, codes: np.ndarray) -> tuple[float, float]:
    """Newton fit of p(y=1|f) = sigmoid(A*f + B) on the training decision values."""
    n_pos = int(codes.sum())
    n_neg = len(codes) - n_pos
    target = np.where(codes == 1, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))
    ridge = 1e-6
    w = np.zeros(2)
    for _ in range(100):
        z = np.clip(w[0] * f + w[1], -350.0, 350.0)
        p = 1.0 / (1.0 + np.exp(-z))
        r = p - target
        g = np.array([r @ f, r.sum()]) + ridge * w
        q = p * (1.0 - p)
        H = np.array([[q @ (f * f) + ridge, q @ f], [q @ f, q.sum() + ridge]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


class SVC(BaseClassifier):
    """Soft-margin Support Vector Classifier trained with SMO. Binary targets only.

    Kernels: 'linear', 'poly' = (gamma*x.y + coef0)**degree, 'rbf' = exp(-gamma*||x-y||**2).
    The n x n kernel matrix is never materialized: SMO reads kernel rows on demand.
    """

    def __init__(self, kernel="linear", C=1.0, degree=3, gamma="scale", coef0=0.0,
                 tol=1e-3, max_passes=100, class_weight=None):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.class_weight = class_weight

    def _sample_C(self, codes: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        if isinstance(self.class_weight, str):
            if self.class_weight != "balanced":
                raise ValueError(f"class_weight tidak dikenal: {self.class_weight!r}")
            counts = np.bincount(codes, minlength=n_classes).astype(float)
            counts[counts == 0] = 1.0
            weights = len(codes) / (n_classes * counts)
        else:
            weights = self._class_weight_vector(self.class_weight, n_classes)
        return self.C * weights[codes]

>>>>>>> 60a0a31 (svm)
    def fit(self, X, y):
        X = self._check_X(X)
        codes = self._encode_y(y)
        if len(self.classes_) != 2:
<<<<<<< HEAD
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
=======
            raise ValueError("SVC hanya mendukung dua kelas")
        if len(X) != len(codes):
            raise ValueError("Jumlah baris X dan y berbeda")
        if self.kernel not in _KERNELS:
            raise ValueError(f"kernel tidak dikenal: {self.kernel!r}")
        if not self.C > 0:
            raise ValueError("C harus positif")
        if self.degree < 1:
            raise ValueError("degree minimal 1")
        if not self.tol > 0:
            raise ValueError("tol harus positif")
        if self.max_passes < 1:
            raise ValueError("max_passes minimal 1")

        n = len(X)
        y_pm = (2.0 * codes - 1.0).astype(float)
        Cvec = self._sample_C(codes)
        gamma = resolve_gamma(self.gamma, X)

        alpha = np.zeros(n)
        f = np.zeros(n)          # decision values incl. bias b
        b = 0.0
        free_eps = 1e-12
        passes = 0
        self._gamma = gamma

        while True:
            margin = y_pm * f
            viol = ((margin < 1.0 - self.tol) & (alpha < Cvec - free_eps)) | \
                   ((margin > 1.0 + self.tol) & (alpha > free_eps))
            if not viol.any():
                break
            if passes >= self.max_passes:
                warnings.warn(
                    f"{type(self).__name__} tidak konvergen setelah {passes} pass "
                    f"(tol={self.tol}); model tetap dipakai, periksa n_iter_",
                    RuntimeWarning, stacklevel=2)
                break
            passes += 1

            for i in np.flatnonzero(viol):
                m_i = y_pm[i] * f[i]
                if not ((m_i < 1.0 - self.tol and alpha[i] < Cvec[i] - free_eps) or
                        (m_i > 1.0 + self.tol and alpha[i] > free_eps)):
                    continue  # fixed by an earlier update in this pass
                E = f - y_pm
                Ei = E[i]
                # heuristic j: |E_j - E_i| among free alphas, else any alpha > 0, else any j
                mask = (alpha > free_eps) & (alpha < Cvec - free_eps)
                mask[i] = False
                if not mask.any():
                    mask = alpha > free_eps
                    mask[i] = False
                if not mask.any():
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                if not mask.any():
                    continue
                cand = np.flatnonzero(mask)
                base = np.abs(E[cand] - Ei)
                Ci = Cvec[i]
                j = int(cand[np.argmax(base)])
                ranked = None
                pos = 0
                tries = 0
                # a frozen pair (opposite labels at opposite bounds) cannot step;
                # retry the next-best j or SMO stalls permanently
                while True:
                    tries += 1
                    K = kernel_matrix(X, X[[i, j]], self.kernel, gamma, self.degree, self.coef0)
                    Kii, Kij, Kjj = K[i, 0], K[j, 0], K[j, 1]
                    eta = Kii + Kjj - 2.0 * Kij
                    aj_new = ai_new = delta_j = delta_i = None
                    if eta > 1e-12:
                        Cj = Cvec[j]
                        if y_pm[i] == y_pm[j]:
                            s = alpha[i] + alpha[j]
                            low, high = max(0.0, s - Ci), min(Cj, s)
                        else:
                            d = alpha[i] - alpha[j]
                            low, high = max(0.0, -d), min(Cj, Ci - d)
                        aj_new = float(np.clip(alpha[j] + y_pm[j] * (Ei - E[j]) / eta, low, high))
                        delta_j = aj_new - alpha[j]
                        delta_i = -y_pm[i] * y_pm[j] * delta_j
                        ai_new = alpha[i] + delta_i
                        if abs(delta_j) >= 1e-12 and 0.0 <= ai_new <= Ci + 1e-9:
                            break
                    if ranked is None:
                        ranked = cand[np.argsort(-base)].tolist()
                    while pos < len(ranked) and ranked[pos] == j:
                        pos += 1
                    if tries >= _MAX_PAIR_TRIES or pos >= len(ranked):
                        j = None
                        break
                    j = ranked[pos]
                    pos += 1
                if j is None:
                    continue

                alpha[i], alpha[j] = ai_new, aj_new

                # bias update: pin the free variable(s) back onto the margin
                Ei_p = f[i] + y_pm[i] * delta_i * Kii + y_pm[j] * delta_j * Kij - y_pm[i]
                Ej_p = f[j] + y_pm[i] * delta_i * Kij + y_pm[j] * delta_j * Kjj - y_pm[j]
                free_i = free_eps < ai_new < Ci - free_eps
                free_j = free_eps < aj_new < Cj - free_eps
                if free_i and free_j:
                    db = -0.5 * (Ei_p + Ej_p)
                elif free_i:
                    db = -Ei_p
                elif free_j:
                    db = -Ej_p
                else:
                    db = 0.0
                b += db

                f += y_pm[i] * delta_i * K[:, 0] + y_pm[j] * delta_j * K[:, 1] + db

        self.alpha_ = alpha
        sv = alpha > 0
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv]
        self._dual_coef = alpha[sv] * y_pm[sv]
        self._b = b
        self.n_iter_ = passes
        self.slack_total = float(np.maximum(0.0, 1.0 - y_pm * f).sum())
        self._platt_A, self._platt_B = _fit_platt(f, codes)
        return self

    def decision_function(self, X) -> np.ndarray:
        self._check_fitted()
        X = self._check_X(X)
        out = np.empty(len(X))
        n_sv = len(self.support_vectors_)
        block = max(1, _DECISION_BLOCK // max(n_sv, 1))
        for start in range(0, len(X), block):
            K = kernel_matrix(X[start:start + block], self.support_vectors_,
                              self.kernel, self._gamma, self.degree, self.coef0)
            out[start:start + block] = K @ self._dual_coef + self._b
        return out

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        f = self.decision_function(X)
        z = np.clip(self._platt_A * f + self._platt_B, -350.0, 350.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])
>>>>>>> 60a0a31 (svm)
