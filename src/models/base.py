from __future__ import annotations

import inspect
import pickle
from pathlib import Path

import numpy as np


class BaseClassifier:
    @classmethod
    def _param_names(cls):
        signature = inspect.signature(cls.__init__)
        return [name for name in signature.parameters if name != "self"]

    def get_params(self) -> dict:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params):
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Parameter tidak dikenal untuk {type(self).__name__}: {name}")
            setattr(self, name, value)
        return self

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    @staticmethod
    def _check_X(X, allow_nan=False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X harus 2 dimensi, didapat bentuk {X.shape}")
        if not allow_nan and np.isnan(X).any():
            raise ValueError("X mengandung NaN; imputasi dulu atau pakai model yang mendukung NaN")
        return X

    def _encode_y(self, y) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y harus 1 dimensi")
        self.classes_, codes = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("y minimal harus memuat dua kelas")
        return codes.astype(np.int64)

    def _check_fitted(self):
        if not hasattr(self, "classes_"):
            raise RuntimeError(f"{type(self).__name__} belum di-fit")

    @staticmethod
    def _class_weight_vector(class_weight, n_classes) -> np.ndarray:
        if class_weight is None:
            return np.ones(n_classes)
        if isinstance(class_weight, dict):
            return np.array([float(class_weight.get(k, 1.0)) for k in range(n_classes)])
        weights = np.asarray(class_weight, dtype=float)
        if weights.shape != (n_classes,):
            raise ValueError("class_weight harus berpanjang sama dengan jumlah kelas")
        return weights

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X, threshold: float | None = None) -> np.ndarray:
        proba = self.predict_proba(X)
        if threshold is not None and proba.shape[1] == 2:
            return self.classes_[(proba[:, 1] >= threshold).astype(int)]
        return self.classes_[np.argmax(proba, axis=1)]

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"Berkas {path} berisi {type(obj).__name__}, bukan {cls.__name__}")
        return obj
