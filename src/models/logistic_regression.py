from __future__ import annotations
import numpy as np
from .base import BaseClassifier


# mogging
def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))

# kill a feature if not significant to prevent overfitting
def soft_threshold(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

class LogisticRegression(BaseClassifier):
    # penalty -> "none", "l1", "l2", "elasticnet"
    # solver -> "sga", "minibatch", "batch", "adam", "newton"
    def __init__(self, penalty = "l2", C=1.0, l1_ratio=0.5, solver="sga", 
                 learning_rate=0.1, decay=0.0, max_iter=1000, batch_size=256, 
                 tol=1e-7, class_weight=None, fit_intercept=True, random_state=42, 
                 record_history=False, history_every=1, beta1=0.9, beta2=0.999):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.solver  = solver 
        self.learning_rate = learning_rate
        self.decay = decay
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.record_history = record_history
        self.history_every = history_every
        self.beta1 = beta1
        self.beta2 = beta2
    
    def _penalty_parts(self, n):
        if self.penalty in (None, "none"):
            return 0.0, 0.0
        lam = 1.0 / (self.C * n)
        rho = {"l2": 0.0, "l1": 1.0, "elasticnet": float(self.l1_ratio)}[self.penalty]
        return lam * (1-rho), lam * rho
    
    def _design(self, X):
            return np.hstack([np.ones((len(X), 1)), X]) if self.fit_intercept else X
    
    # log likelihood
    def _objective(self, A, y, sw, b, lam2, lam1):
        z = A @ b
        ll = np.sum(sw * (y * z - np.logaddexp(0.0, z))) / len(y)
        w = b[1:] if self.fit_intercept else b
        return ll - lam2 / 2 * np.dot(w, w) - lam1 * np.abs(w).sum()

    def _mask(self, b):
        m = np.ones_like(b)
        if self.fit_intercept:
            m[0] = 0.0
        return m
    
    def fit(self, X, y, sample_weight=None):
            X = self._check_X(X)
            y = self._encode_y(y).astype(float)
            if len(self.classes_) != 2:
                raise ValueError("LogisticRegression ini hanya untuk klasifikasi biner")
            sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
            sw = sw * self._class_weight_vector(self.class_weight, 2)[y.astype(int)]
            
            A = self._design(X)
            n = len(y)
            lam2, lam1 = self._penalty_parts(n)
            b = np.zeros(A.shape[1])
            self.history_ = {"params": [b.copy()], "objective": [self._objective(A, y, sw, b, lam2, lam1)]}
            
            solver = getattr(self, f"_fit_{self.solver}", None)
            if solver is None:
                raise ValueError(f"idk what this solver is cuh: {self.solver}")
            b = solver(A, y, sw, b, lam2, lam1)
            self._set_coef(b)
            if not self.record_history:
                self.history_ = {"objective": self.history_["objective"]}
            return self
    
    def _set_coef(self, b):
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(b[0]), b[1:].copy()
        else:
            self.intercept_, self.coef_ = 0.0, b.copy()
    
    def _record(self, step, A, y, sw, b, lam2, lam1):
            if step % self.history_every == 0:
                self.history_["objective"].append(self._objective(A, y, sw, b, lam2, lam1))
                if self.record_history:
                    self.history_["params"].append(b.copy())
    
    # if difference between obj of current and previous hella tiny just stop. < self.tol
    def _converged(self):
        obj = self.history_["objective"]
        return len(obj) >= 2 and abs(obj[-1] - obj[-2]) < self.tol
    
    # gradient algorithms
    def _gradient_loop(self, A, y, sw, b, lam2, lam1, batch, use_adam=False):
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        mask = self._mask(b)
        m_t = np.zeros_like(b)
        v_t = np.zeros_like(b)
        t =0
        self.n_iter_ = 0
        
        for epoch in range(self.max_iter):
            order = rng.permutation(n) if batch < n else np.arange(n)
            for start in range(0, n, batch):
                rows = order[start:start + batch]
                Ab, yb, swb = A[rows], y[rows], sw[rows]
                grad = Ab.T @ (swb * (yb - sigmoid(Ab @ b))) / len(rows)
                grad -= lam2 * mask * b
                t += 1
                if use_adam:
                    m_t = self.beta1 * m_t + (1 - self.beta1) * grad
                    v_t = self.beta2 * v_t + (1 - self.beta2) * grad**2
                    m_hat = m_t / (1 - self.beta1**t)
                    v_hat = v_t / (1 - self.beta2**t)
                    eta = self.learning_rate
                    step = eta * m_hat / (np.sqrt(v_hat) + 1e-8)
                    b = b + step
                    if lam1 > 0:
                        # Prox dengan langkah per-koordinat efektif Adam.
                        eff = eta / (np.sqrt(v_hat) + 1e-8)
                        b = np.where(mask > 0, soft_threshold(b, eff * lam1), b)
                else:
                    eta = self.learning_rate / (1.0 + self.decay * t)
                    b = b + eta * grad
                    if lam1 > 0:
                        b = np.where(mask > 0, soft_threshold(b, eta * lam1), b)
                if self.record_history and batch < n:
                    self.history_["params"].append(b.copy())
            self.n_iter_ = epoch + 1
            obj = self._objective(A, y, sw, b, lam2, lam1)
            self.history_["objective"].append(obj)
            if self.record_history and batch >= n:
                self.history_["params"].append(b.copy())
            if self._converged():
                break
        return b
    
    def _fit_sga(self, A, y, sw, b, lam2, lam1):
            return self._gradient_loop(A, y, sw, b, lam2, lam1, batch=1)
    
    def _fit_minibatch(self, A, y, sw, b, lam2, lam1):
        return self._gradient_loop(A, y, sw, b, lam2, lam1, batch=self.batch_size)

    def _fit_batch(self, A, y, sw, b, lam2, lam1):
        return self._gradient_loop(A, y, sw, b, lam2, lam1, batch=len(y))

    def _fit_adam(self, A, y, sw, b, lam2, lam1):
        return self._gradient_loop(A, y, sw, b, lam2, lam1, batch=self.batch_size, use_adam=True)
    
    # newton algorithm
    def _fit_newton(self, A, y, sw, b, lam2, lam1):
        n = len(y)
        mask = self._mask(b)
        self.n_iter_ = 0
        for it in range(self.max_iter):
            p = sigmoid(A @ b)
            grad = A.T @ (sw * (y - p)) / n - lam2 * mask * b
            hess = (A * (sw * p * (1 - p))[:, None]).T @ A / n + lam2 * np.diag(mask)
            hess += 1e-10 * np.eye(len(b))
            if lam1 == 0:
                direction = np.linalg.solve(hess, grad)
            else:
                direction = self._prox_newton_direction(hess, grad, b, mask, lam1)
            
            # backtracking
            obj0 = self._objective(A, y, sw, b, lam2, lam1)
            step = 1.0
            while step > 1e-8:
                candidate = b + step * direction
                if self._objective(A, y, sw, candidate, lam2, lam1) >= obj0 - 1e-15:
                    break
                step *= 0.5
            b = candidate
            self.n_iter_ = it + 1
            self._record(it + 1, A, y, sw, b, lam2, lam1)
            if self._converged() or np.max(np.abs(step * direction)) < 1e-10:
                break
        return b

    
    @staticmethod
    def _prox_newton_direction(H, g, b, mask, lam1, sweeps=100):
        d = np.zeros_like(b)
        Hd = np.zeros_like(b)
        for _ in range(sweeps):
            max_change = 0.0
            for j in range(len(b)):
                hjj = H[j, j]
                rest = Hd[j] - hjj * d[j]
                if mask[j] == 0:
                    new = (g[j] - rest) / hjj
                else:
                    u = b[j] + (g[j] - rest) / hjj
                    new = soft_threshold(u, lam1 / hjj) - b[j]
                delta = new - d[j]
                if delta != 0.0:
                    Hd += H[:, j] * delta
                    d[j] = new
                    max_change = max(max_change, abs(delta))
            if max_change < 1e-12:
                break
        return d
    
    
    
    def predict_proba(self, X):
        self._check_fitted()
        X = self._check_X(X)
        p = sigmoid(X @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p, p])