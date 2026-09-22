"""Leakage-safe preprocessing.

Every step that learns something from data (imputer medians, scaler mean/std) lives INSIDE a
scikit-learn Pipeline, so during cross-validation it is re-fitted on each training fold only.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES, ZERO_IS_MISSING

ZERO_IDX = [FEATURES.index(c) for c in ZERO_IS_MISSING]


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Stateless: turns physiologically impossible zeros into NaN (no statistics learned)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        for c in (self.columns if self.columns is not None else ZERO_IDX):
            X[X[:, c] == 0, c] = np.nan
        return X


def build_pipeline(estimator, scale=False, treat_zeros=True):
    """zero->NaN -> median imputation (train-fold only) -> optional StandardScaler -> classifier."""
    steps = []
    if treat_zeros:
        steps += [("zero_to_nan", ZeroToNaN()),
                  ("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)
