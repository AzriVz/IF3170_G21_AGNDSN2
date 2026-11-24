from __future__ import annotations

from .models import CARTClassifier, KNNClassifier, LogisticRegression, NaiveBayes, SVC
from .models import TreeAugmentedNaiveBayes
from .preprocessing import make_pipeline

SEED = 42

KNN_WEIGHTS = {
    "days_before_arrival": 3.0,
    "arrival_ord": 2.0,
    "special_requests": 2.0,
    "log_price": 2.0,
    "booking_ord": 2.0,
    "nightly_rate_idr": 1.5,
    "booking_channel=Online Travel Agent": 1.0,
    "booking_channel=Offline Agent": 1.0,
    "booking_channel=Corporate": 1.0,
    "total_nights": 1.0,
    "needs_parking": 1.0,
    "is_returning_guest": 1.0,
    "arrival_day": 1.0,
    "num_adults": 0.5,
    "weekend_nights": 0.5,
    "room_category=Type C": 0.5,
    "log_lead": 0.5,
}
# source (distance_power=2): Shepard (1968), "A Two-Dimensional Interpolation Function for Irregularly-Spaced Data", Proc. ACM National Conference.
KNN_PARAMS = {"n_neighbors": 30, "metric": "manhattan", "weights": "distance", "distance_power": 2.0}
KNN_THRESHOLD = 0.48
KNN_BASELINE_PARAMS = {"n_neighbors": 5, "metric": "euclidean", "weights": "uniform"}
CART_PARAMS = {"criterion": "gini", "max_depth": 12, "min_samples_leaf": 1, "ccp_alpha": 2e-4, "missing": "learned"}
CART_THRESHOLD = 0.52
LR_BIN_COLUMNS = ["days_before_arrival", "nightly_rate_idr", "arrival_ord", "booking_ord", "total_nights", "price_per_guest", "arrival_day"]
LR_PARAMS = {"penalty": "l2", "C": 3.0, "solver": "newton"}
LR_THRESHOLD = 0.44
SVM_SUBSAMPLE = 6000
SVM_PARAMS = {
    "linear": {"kernel": "linear", "C": 1.0},
    "poly": {"kernel": "poly", "C": 3.0, "degree": 3, "gamma": "scale", "coef0": 1.0},
    "rbf": {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
}
SVM_THRESHOLDS = {"linear": 0.43, "poly": 0.41, "rbf": 0.43}
NB_PARAMS = {"continuous": "discretize", "n_bins": 20}
TAN_PARAMS = {"n_bins": 20}
KNN_IMPUTE = ("days_before_arrival", "nightly_rate_idr")

def knn_pipeline(weights=None, knn_impute=KNN_IMPUTE):
    weights = KNN_WEIGHTS if weights is None else weights
    return make_pipeline("distance", columns=list(weights), weights=weights, indicators=("special_requests",), knn_impute=knn_impute)

def knn_model(**overrides):
    return KNNClassifier(**{**KNN_PARAMS, **overrides})

def knn_baseline_pipeline():
    return make_pipeline("distance", indicators=(), derived=[])

def cart_pipeline():
    return make_pipeline("tree", impute="median", indicators=("special_requests",))

def cart_model(**overrides):
    return CARTClassifier(**{**CART_PARAMS, **overrides})

def lr_pipeline():
    return make_pipeline("distance", bins=LR_BIN_COLUMNS, n_bins=20)

def lr_model(**overrides):
    return LogisticRegression(**{**LR_PARAMS, **overrides})

def svm_pipeline():
    return knn_pipeline()


def svm_model(kernel="rbf", **overrides):
    return SVC(**{**SVM_PARAMS[kernel], **overrides})

def nb_pipeline():
    return make_pipeline("bayes", impute="none", derived=[], indicators=())

def nb_model(**overrides):
    return NaiveBayes(**{**NB_PARAMS, **overrides})

def tan_model(**overrides):
    return TreeAugmentedNaiveBayes(**{**TAN_PARAMS, **overrides})