from __future__ import annotations

import numpy as np
import pandas as pd

from .models.knn import elementwise_sq_distances, smallest_k_indices

TARGET = "is_canceled"
ID_COLUMN = "id"
CATEGORICAL = ["meal_package", "room_category", "booking_channel"]
NUMERIC = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights",
    "needs_parking", "days_before_arrival", "arrival_year", "arrival_month",
    "arrival_day", "is_returning_guest", "prior_cancellations",
    "prior_completed_bookings", "nightly_rate_idr", "special_requests",
]
RAW_FEATURES = NUMERIC + CATEGORICAL

DATE_ORIGIN = pd.Timestamp("2023-01-01")


class Step:
    def fit(self, df: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df, y=None):
        return self.fit(df, y).transform(df)


class DataCleaner(Step):
    def __init__(self, zero_price_to_nan=False, zero_adults_to_nan=False, max_children=3):
        self.zero_price_to_nan = zero_price_to_nan
        self.zero_adults_to_nan = zero_adults_to_nan
        self.max_children = max_children

    def transform(self, df):
        out = df.copy()
        if self.zero_price_to_nan:
            bad = (out["nightly_rate_idr"] == 0) & (out["booking_channel"] != "Complimentary")
            out.loc[bad, "nightly_rate_idr"] = np.nan
        if self.zero_adults_to_nan:
            out.loc[out["num_adults"] == 0, "num_adults"] = np.nan
        if self.max_children is not None:
            out["num_children"] = out["num_children"].clip(upper=self.max_children)
        return out


class FeatureBuilder(Step):
    def __init__(self, features=None):
        self.features = features

    AVAILABLE = (
        "arrival_ord", "arrival_dow", "booking_ord", "total_nights", "total_guests",
        "weekend_ratio", "total_cost", "price_per_guest", "log_lead", "log_price",
        "month_sin", "month_cos", "has_prior_cancel",
    )

    def transform(self, df):
        out = df.copy()
        wanted = set(self.AVAILABLE if self.features is None else self.features)
        date = pd.to_datetime(
            dict(year=out["arrival_year"], month=out["arrival_month"], day=out["arrival_day"]),
            errors="coerce",
        )
        arrival_ord = (date - DATE_ORIGIN).dt.days.astype(float)
        nights = out["weekend_nights"] + out["weekday_nights"]
        guests = out["num_adults"] + out["num_children"]
        built = {
            "arrival_ord": arrival_ord,
            "arrival_dow": date.dt.dayofweek.astype(float),
            "booking_ord": arrival_ord - out["days_before_arrival"],
            "total_nights": nights.astype(float),
            "total_guests": guests,
            "weekend_ratio": out["weekend_nights"] / nights.where(nights > 0),
            "total_cost": out["nightly_rate_idr"] * nights,
            "price_per_guest": out["nightly_rate_idr"] / guests.where(guests > 0),
            "log_lead": np.log1p(out["days_before_arrival"]),
            "log_price": np.log1p(out["nightly_rate_idr"]),
            "month_sin": np.sin(2 * np.pi * out["arrival_month"] / 12),
            "month_cos": np.cos(2 * np.pi * out["arrival_month"] / 12),
            "has_prior_cancel": (out["prior_cancellations"] > 0).astype(float),
        }
        for name in self.AVAILABLE:
            if name in wanted:
                out[name] = built[name]
        return out


class Imputer(Step):
    DEFAULT_GROUPS = {
        "nightly_rate_idr": [
            ["room_category", "arrival_year", "arrival_month"],
            ["room_category", "arrival_year"],
            ["room_category"],
        ],
    }

    def __init__(self, strategy="median", indicators=(), group_keys=None):
        self.strategy = strategy
        self.indicators = tuple(indicators)
        self.group_keys = group_keys

    def fit(self, df, y=None):
        self.fill_ = {}
        for column in df.columns:
            if df[column].dtype == object:
                mode = df[column].mode(dropna=True)
                self.fill_[column] = mode.iloc[0] if len(mode) else "missing"
            else:
                self.fill_[column] = float(df[column].median())
        self.group_fill_ = {}
        if self.strategy == "group":
            groups = self.DEFAULT_GROUPS if self.group_keys is None else self.group_keys
            for column, levels in groups.items():
                tables = []
                for keys in levels:
                    table = df.groupby(keys, dropna=True)[column].median().dropna()
                    tables.append((keys, table))
                self.group_fill_[column] = tables
        return self

    def transform(self, df):
        out = df.copy()
        for column in self.indicators:
            out[f"{column}_missing"] = df[column].isna().astype(float)
        if self.strategy == "none":
            return out
        for column, tables in self.group_fill_.items():
            for keys, table in tables:
                missing = out[column].isna()
                if not missing.any():
                    break
                lookup = pd.MultiIndex.from_frame(out.loc[missing, keys]) if len(keys) > 1 else out.loc[missing, keys[0]]
                values = table.reindex(lookup).to_numpy()
                out.loc[missing, column] = values
        for column, value in self.fill_.items():
            if column in out:
                out[column] = out[column].fillna(value)
        return out


# source: Troyanskaya et al. (2001), "Missing Value Estimation Methods for DNA Microarrays", Bioinformatics 17(6).
class KNNRegressionImputer(Step):
    NUMERIC_CONTEXT = ["arrival_ord", "days_before_arrival", "nightly_rate_idr", "special_requests",
                       "total_nights", "num_adults", "num_children", "needs_parking", "is_returning_guest"]
    CATEGORICAL_CONTEXT = ["booking_channel", "room_category", "meal_package"]

    def __init__(self, targets=("days_before_arrival", "nightly_rate_idr"), n_neighbors=15,
                 cat_weight=2.0, batch_size=512):
        self.targets = tuple(targets)
        self.n_neighbors = n_neighbors
        self.cat_weight = cat_weight
        self.batch_size = batch_size

    @staticmethod
    def _with_helpers(df):
        df = df.copy()
        date = pd.to_datetime(dict(year=df["arrival_year"], month=df["arrival_month"],
                                   day=df["arrival_day"]), errors="coerce")
        df["arrival_ord"] = (date - DATE_ORIGIN).dt.days.astype(float)
        df["total_nights"] = (df["weekend_nights"] + df["weekday_nights"]).astype(float)
        return df

    def _design(self, df, exclude):
        cols = [c for c in self.NUMERIC_CONTEXT if c != exclude]
        X = df[cols].astype(float).fillna(self.median_[cols])
        blocks = [((X - self.mean_[cols]) / self.std_[cols]).to_numpy()]
        for c in self.CATEGORICAL_CONTEXT:
            if c != exclude:
                levels = np.array(self.levels_[c], dtype=object)
                blocks.append(self.cat_weight * (df[c].to_numpy()[:, None] == levels[None, :]).astype(float))
        return np.hstack(blocks)

    def fit(self, df, y=None):
        df = self._with_helpers(df)
        X = df[self.NUMERIC_CONTEXT].astype(float)
        self.median_, self.mean_ = X.median(), X.mean()
        self.std_ = X.std().replace(0, 1.0)
        self.levels_ = {c: sorted(df[c].dropna().unique()) for c in self.CATEGORICAL_CONTEXT}
        self.reference_ = {}
        for target in self.targets:
            known = df[target].notna().to_numpy()
            self.reference_[target] = (self._design(df[known], target),
                                       df.loc[known, target].to_numpy(dtype=float))
        return self

    def transform(self, df):
        out = df.copy()
        context = self._with_helpers(df)
        for target in self.targets:
            missing = context[target].isna().to_numpy()
            if not missing.any():
                continue
            R, values = self.reference_[target]
            Q = self._design(context[missing], target)
            filled = np.empty(len(Q))
            k = min(self.n_neighbors, len(R))
            for start in range(0, len(Q), self.batch_size):
                dist = elementwise_sq_distances(Q[start:start + self.batch_size], R)
                nearest = smallest_k_indices(dist, k)
                filled[start:start + self.batch_size] = np.median(values[nearest], axis=1)
            out.loc[missing, target] = filled
        return out


class CategoryEncoder(Step):
    def __init__(self, mode="onehot", min_frequency=20, columns=CATEGORICAL):
        self.mode = mode
        self.min_frequency = min_frequency
        self.columns = list(columns)

    def fit(self, df, y=None):
        self.categories_ = {}
        for column in self.columns:
            counts = df[column].value_counts(dropna=True)
            frequent = sorted(counts[counts >= self.min_frequency].index.tolist())
            rare = counts[counts < self.min_frequency].index.tolist()
            self.categories_[column] = (frequent, set(rare))
        return self

    def transform(self, df):
        out = df.copy()
        for column in self.columns:
            frequent, rare = self.categories_[column]
            values = out[column].where(~out[column].isin(rare), "other")
            levels = frequent + (["other"] if rare else [])
            if self.mode == "onehot":
                for level in levels:
                    out[f"{column}={level}"] = (values == level).astype(float)
            else:
                code = {level: float(i) for i, level in enumerate(levels)}
                out[column + "_code"] = values.map(code).astype(float)
            out = out.drop(columns=column)
        return out

    def ordinal_columns(self):
        return [c + "_code" for c in self.columns] if self.mode == "ordinal" else []


class QuantileBinner(Step):
    def __init__(self, columns=(), n_bins=10):
        self.columns = tuple(columns)
        self.n_bins = n_bins

    def fit(self, df, y=None):
        self.edges_ = {}
        for column in self.columns:
            values = df[column].dropna().to_numpy(dtype=float)
            qs = np.quantile(values, np.linspace(0, 1, self.n_bins + 1)[1:-1])
            self.edges_[column] = np.unique(qs)
        return self

    def transform(self, df):
        blocks = {}
        for column, edges in self.edges_.items():
            codes = np.searchsorted(edges, df[column].to_numpy(dtype=float), side="right")
            for b in range(1, len(edges) + 1):
                blocks[f"{column}_bin{b}"] = (codes == b).astype(float)
        return pd.concat([df, pd.DataFrame(blocks, index=df.index)], axis=1)


class Scaler(Step):
    def __init__(self, method="standard", exclude=()):
        self.method = method
        self.exclude = tuple(exclude)

    def fit(self, df, y=None):
        columns = [c for c in df.columns if c not in self.exclude]
        values = df[columns].astype(float)
        if self.method == "standard":
            center, scale = values.mean(), values.std(ddof=0)
        elif self.method == "robust":
            center = values.median()
            scale = values.quantile(0.75) - values.quantile(0.25)
            scale = scale.where(scale > 0, values.std(ddof=0))
        elif self.method == "minmax":
            center, scale = values.min(), values.max() - values.min()
        else:
            raise ValueError(f"Metode scaling tidak dikenal: {self.method}")
        self.center_ = center
        self.scale_ = scale.where(scale > 0, 1.0)
        return self

    def transform(self, df):
        out = df.copy()
        columns = list(self.center_.index)
        out[columns] = (out[columns].astype(float) - self.center_) / self.scale_
        return out


class ColumnSelector(Step):
    def __init__(self, columns=None, weights=None):
        self.columns = columns
        self.weights = weights

    def fit(self, df, y=None):
        self.columns_ = list(df.columns) if self.columns is None else list(self.columns)
        missing = [c for c in self.columns_ if c not in df.columns]
        if missing:
            raise KeyError(f"Kolom tidak ada: {missing}")
        return self

    def transform(self, df):
        out = df[self.columns_].copy()
        if self.weights is not None:
            for column, weight in self.weights.items():
                if column in out:
                    out[column] = out[column] * weight
        return out


class PCA(Step):
    def __init__(self, n_components=None, prefix="pc"):
        self.n_components = n_components
        self.prefix = prefix

    def fit(self, df, y=None):
        X = df.to_numpy(dtype=float)
        self.mean_ = X.mean(axis=0)
        _, singular, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
        vt = vt * signs[:, None]
        variance = singular**2 / max(len(X) - 1, 1)
        self.explained_variance_ratio_full_ = variance / variance.sum()
        k = vt.shape[0] if self.n_components is None else self.n_components
        if isinstance(k, float) and 0 < k < 1:
            k = int(np.searchsorted(np.cumsum(self.explained_variance_ratio_full_), k) + 1)
        self.components_ = vt[:k]
        self.explained_variance_ratio_ = self.explained_variance_ratio_full_[:k]
        return self

    def transform(self, df):
        Z = (df.to_numpy(dtype=float) - self.mean_) @ self.components_.T
        names = [f"{self.prefix}{i + 1}" for i in range(Z.shape[1])]
        return pd.DataFrame(Z, index=df.index, columns=names)


def drop_duplicate_rows(df: pd.DataFrame, y: np.ndarray, conflicting="keep"):
    features = [c for c in df.columns if c != ID_COLUMN]
    frame = df[features].copy()
    frame["__y"] = np.asarray(y)
    keep = ~frame.duplicated(keep="first")
    if conflicting == "drop":
        n_labels = frame.groupby(features, dropna=False)["__y"].transform("nunique")
        keep &= n_labels.to_numpy() == 1
    keep = keep.to_numpy()
    return df.loc[keep], np.asarray(y)[keep]


def random_oversample(X: np.ndarray, y: np.ndarray, random_state=42):
    rng = np.random.default_rng(random_state)
    labels, counts = np.unique(y, return_counts=True)
    target = counts.max()
    index = [np.arange(len(y))]
    for label, count in zip(labels, counts):
        if count < target:
            members = np.flatnonzero(y == label)
            index.append(rng.choice(members, size=target - count, replace=True))
    index = np.concatenate(index)
    return X[index], y[index]


class PreprocessingPipeline:
    def __init__(self, steps, drop_duplicates=False, oversample=False, random_state=42):
        self.steps = list(steps)
        self.drop_duplicates = drop_duplicates
        self.oversample = oversample
        self.random_state = random_state

    def _select_raw(self, df):
        return df[RAW_FEATURES].copy()

    def fit(self, df, y=None):
        self.fit_resample(df, y)
        return self

    def fit_resample(self, df, y):
        y = np.asarray(y)
        if self.drop_duplicates:
            df, y = drop_duplicate_rows(df[RAW_FEATURES], y, conflicting=self.drop_duplicates)
        frame = self._select_raw(df)
        for step in self.steps:
            frame = step.fit_transform(frame, y)
        self.feature_names_ = list(frame.columns)
        X = frame.to_numpy(dtype=float)
        if self.oversample:
            X, y = random_oversample(X, y, self.random_state)
        return X, y

    def fit_transform(self, df, y):
        return self.fit_resample(df, y)[0]

    def transform(self, df) -> np.ndarray:
        frame = self._select_raw(df)
        for step in self.steps:
            frame = step.transform(frame)
        return frame[self.feature_names_].to_numpy(dtype=float)

    @property
    def categorical_indices_(self):
        encoders = [s for s in self.steps if isinstance(s, CategoryEncoder)]
        names = set(c for e in encoders for c in e.ordinal_columns())
        return [i for i, name in enumerate(self.feature_names_) if name in names]

    @property
    def fit_kwargs(self):
        idx = self.categorical_indices_
        return {"categorical_features": idx} if idx else {}


def make_pipeline(kind="distance", columns=None, weights=None, impute="median",
                  indicators=("special_requests",), scale="standard",
                  derived=None, onehot_min_frequency=20, clean=None,
                  drop_duplicates=False, oversample=False, pca=None, bins=None, n_bins=10,
                  knn_impute=()):
    steps = [DataCleaner(**(clean or {}))]
    if knn_impute:
        steps.append(KNNRegressionImputer(knn_impute))
    steps.append(FeatureBuilder(derived))
    steps.append(Imputer(impute, indicators=indicators))
    if bins:
        steps.append(QuantileBinner(bins, n_bins))
    if kind == "distance":
        steps.append(CategoryEncoder("onehot", min_frequency=onehot_min_frequency))
        if scale:
            steps.append(Scaler(scale))
    elif kind in ("tree", "bayes"):
        steps.append(CategoryEncoder("ordinal", min_frequency=onehot_min_frequency))
    else:
        raise ValueError(f"Jenis pipeline tidak dikenal: {kind}")
    if columns is not None or weights is not None:
        steps.append(ColumnSelector(columns, weights))
    if pca is not None:
        steps.append(PCA(pca))
    return PreprocessingPipeline(steps, drop_duplicates=drop_duplicates, oversample=oversample)
