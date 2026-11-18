

from __future__ import annotations

import numpy as np

from ._validation import check_array, check_is_fitted
from .stats import StreamingQuantile, StreamingStats

__all__ = ["StandardScaler", "MinMaxScaler", "SimpleImputer", "OneHotEncoder"]


class _Transformer:

    def fit(self, X, y=None):
        self.reset()
        return self.partial_fit(X, y)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def partial_fit_transform(self, X, y=None):
        return self.partial_fit(X, y).transform(X)

    def _check_width(self, X, attr="n_features_in_"):
        expected = getattr(self, attr, None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"{type(self).__name__} was fitted with {expected} features but "
                f"got X with {X.shape[1]}."
            )

    def __repr__(self):
        return f"{type(self).__name__}()"


class StandardScaler(_Transformer):
    

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.reset()

    def reset(self):
        self._stats = StreamingStats()
        self.n_features_in_ = None
        self.mean_ = self.var_ = self.scale_ = None
        return self

    def partial_fit(self, X, y=None):
        X = check_array(X)
        self._check_width(X)
        self._stats.update_stats(X)
        self.n_features_in_ = X.shape[1]
        mean = self._stats.mean
        var = self._stats.var()
        self.mean_ = np.where(np.isnan(mean), 0.0, mean)
        self.var_ = np.where(np.isnan(var), 0.0, var)
        std = np.sqrt(self.var_)
        self.scale_ = np.where(std > np.finfo(float).eps * np.maximum(1.0, np.abs(self.mean_)),
                               std, 1.0)
        return self

    @property
    def n_samples_seen_(self):
        return self._stats.count_

    def transform(self, X):
        check_is_fitted(self, "mean_", "transform()")
        X = check_array(X)
        self._check_width(X)
        out = X - self.mean_ if self.with_mean else X.copy()
        if self.with_std:
            out /= self.scale_
        return out

    def inverse_transform(self, Z):
        check_is_fitted(self, "mean_", "inverse_transform()")
        Z = check_array(Z, name="Z")
        self._check_width(Z)
        out = Z * self.scale_ if self.with_std else Z.copy()
        return out + self.mean_ if self.with_mean else out


class MinMaxScaler(_Transformer):
    

    def __init__(self, feature_range=(0.0, 1.0), clip: bool = False):
        lo, hi = feature_range
        if not hi > lo:
            raise ValueError(f"feature_range must be (min, max) with max > min, got {feature_range}.")
        self.feature_range = (float(lo), float(hi))
        self.clip = clip
        self.reset()

    def reset(self):
        self._stats = StreamingStats()
        self.n_features_in_ = None
        self.data_min_ = self.data_max_ = None
        return self

    def partial_fit(self, X, y=None):
        X = check_array(X)
        self._check_width(X)
        self._stats.update_stats(X)
        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.where(np.isnan(self._stats.min_), 0.0, self._stats.min_)
        self.data_max_ = np.where(np.isnan(self._stats.max_), 0.0, self._stats.max_)
        return self

    def transform(self, X):
        check_is_fitted(self, "data_min_", "transform()")
        X = check_array(X)
        self._check_width(X)
        lo, hi = self.feature_range
        span = self.data_max_ - self.data_min_
        safe = np.where(span > 0, span, 1.0)
        out = (X - self.data_min_) / safe * (hi - lo) + lo
        out = np.where(span > 0, out, np.where(np.isnan(X), np.nan, lo))
        return np.clip(out, lo, hi) if self.clip else out


class SimpleImputer(_Transformer):
   

    _STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0,
                 reservoir_size: int = 10_000, random_state=None):
        if strategy not in self._STRATEGIES:
            raise ValueError(f"strategy must be one of {self._STRATEGIES}, got {strategy!r}.")
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.reservoir_size = reservoir_size
        self.random_state = random_state
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.statistics_ = None
        self.empty_columns_ = np.array([], dtype=int)
        self._stats = StreamingStats()
        self._quant = StreamingQuantile(self.reservoir_size, self.random_state)
        self._counts = None
        return self

    def _update_counts(self, X):
        if self._counts is None:
            self._counts = [dict() for _ in range(X.shape[1])]
        for j in range(X.shape[1]):       
            col = X[:, j]
            vals, cnt = np.unique(col[~np.isnan(col)], return_counts=True)
            d = self._counts[j]
            for v, c in zip(vals.tolist(), cnt.tolist()):
                d[v] = d.get(v, 0) + c

    def _mode(self):
        out = np.full(self.n_features_in_, np.nan)
        for j, d in enumerate(self._counts):
            if d:
                best = max(d.values())
                out[j] = min(v for v, c in d.items() if c == best)
        return out

    def partial_fit(self, X, y=None):
        X = check_array(X)
        self._check_width(X)
        self.n_features_in_ = X.shape[1]
        if self.strategy == "mean":
            stat = self._stats.update_stats(X).mean
        elif self.strategy == "median":
            stat = self._quant.update_stats(X).quantile(0.5)
        elif self.strategy == "most_frequent":
            self._update_counts(X)
            stat = self._mode()
        else:
            stat = np.full(X.shape[1], self.fill_value)
        stat = np.asarray(stat, dtype=float)
        self.empty_columns_ = np.flatnonzero(np.isnan(stat))
        self.statistics_ = np.where(np.isnan(stat), self.fill_value, stat)
        return self

    def transform(self, X):
        check_is_fitted(self, "statistics_", "transform()")
        X = check_array(X)
        self._check_width(X)
        return np.where(np.isnan(X), self.statistics_, X)


class OneHotEncoder(_Transformer):
    

    _MISSING = frozenset({"", "nan", "none", "null"})

    def __init__(self, handle_unknown: str = "ignore", max_categories: int | None = None):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self.reset()

    def reset(self):
        self.n_features_in_ = None
        self.categories_ = None
        self._slot = None          
        self.n_features_out_ = 0
        return self

    @staticmethod
    def _as_str(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        return X.astype(str)

    def partial_fit(self, X, y=None):
        X = self._as_str(X)
        self._check_width(X)
        if self._slot is None:
            self.n_features_in_ = X.shape[1]
            self._slot = [dict() for _ in range(X.shape[1])]
            self.categories_ = [[] for _ in range(X.shape[1])]
        for j in range(X.shape[1]):
            col = X[:, j]
            vals, first = np.unique(col, return_index=True)
            for v in vals[np.argsort(first)].tolist():
                if v.strip().lower() in self._MISSING or v in self._slot[j]:
                    continue
                if self.max_categories is not None and len(self._slot[j]) >= self.max_categories:
                    continue
                self._slot[j][v] = self.n_features_out_
                self.categories_[j].append(v)
                self.n_features_out_ += 1
        return self

    def transform(self, X):
        check_is_fitted(self, "_slot", "transform()")
        X = self._as_str(X)
        self._check_width(X)
        out = np.zeros((X.shape[0], self.n_features_out_))
        rows = np.arange(X.shape[0])
        for j in range(X.shape[1]):
            vals, inv = np.unique(X[:, j], return_inverse=True)
            lut = np.array([self._slot[j].get(v, -1) for v in vals.tolist()], dtype=int)
            if self.handle_unknown == "error":
                bad = [v for v, s in zip(vals.tolist(), lut) if s < 0
                       and v.strip().lower() not in self._MISSING]
                if bad:
                    raise ValueError(f"Unknown categories {bad} in column {j}.")
            cols = lut[inv.ravel()]
            hit = cols >= 0
            out[rows[hit], cols[hit]] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "_slot", "get_feature_names_out()")
        names = input_features or [f"x{j}" for j in range(self.n_features_in_)]
        out = [None] * self.n_features_out_
        for j, d in enumerate(self._slot):
            for v, s in d.items():
                out[s] = f"{names[j]}={v}"
        return out
