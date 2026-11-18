

from __future__ import annotations

import warnings
from collections import deque
from typing import Iterable

import numpy as np

from ._validation import check_array, check_positive_int

__all__ = [
    "StreamingStats",
    "StreamingQuantile",
    "StreamingHistogram",
    "update_stats",
    "chunked_mean",
    "chunked_variance",
    "chunked_quantile",
    "chunked_histogram",
]


class StreamingStats:
    


    def __init__(self):
        self.reset()

    def reset(self) -> "StreamingStats":
        self.n_features_ = None
        self.count_ = None
        self.mean_ = None
        self._m2 = None
        self.min_ = None
        self.max_ = None
        self.n_chunks_ = 0
        return self

    def _init(self, d: int) -> None:
        self.n_features_ = d
        self.count_ = np.zeros(d, dtype=np.int64)
        self.mean_ = np.full(d, np.nan)
        self._m2 = np.zeros(d)
        self.min_ = np.full(d, np.nan)
        self.max_ = np.full(d, np.nan)

    def update_stats(self, X_chunk) -> "StreamingStats":
        
        X = check_array(X_chunk, name="X_chunk")
        if self.n_features_ is None:
            self._init(X.shape[1])
        elif X.shape[1] != self.n_features_:
            raise ValueError(
                f"X_chunk has {X.shape[1]} features but earlier chunks had "
                f"{self.n_features_}."
            )
        valid = ~np.isnan(X)
        n_b = valid.sum(axis=0)
        has = n_b > 0
        safe_n = np.where(has, n_b, 1)
        mean_b = np.where(valid, X, 0.0).sum(axis=0) / safe_n
        m2_b = (np.where(valid, X - mean_b, 0.0) ** 2).sum(axis=0)

        n_a = self.count_
        n = n_a + n_b
        mean_a = np.where(n_a > 0, self.mean_, 0.0)
        delta = mean_b - mean_a
        safe_tot = np.where(n > 0, n, 1)
        new_mean = mean_a + delta * n_b / safe_tot
        new_m2 = self._m2 + m2_b + delta ** 2 * n_a * n_b / safe_tot

        self.mean_ = np.where(n > 0, new_mean, np.nan)
        self._m2 = np.where(has, new_m2, self._m2)
        self.count_ = n
        if has.any():
            with np.errstate(invalid="ignore"):
                cmin = np.where(has, np.nanmin(np.where(valid, X, np.inf), axis=0), np.nan)
                cmax = np.where(has, np.nanmax(np.where(valid, X, -np.inf), axis=0), np.nan)
            self.min_ = np.fmin(self.min_, cmin)
            self.max_ = np.fmax(self.max_, cmax)
        self.n_chunks_ += 1
        return self

    update = update_stats
    partial_fit = update_stats

    def merge(self, other: "StreamingStats") -> "StreamingStats":
        if other.n_features_ is None:
            return self
        if self.n_features_ is None:
            self._init(other.n_features_)
        if other.n_features_ != self.n_features_:
            raise ValueError("Cannot merge statistics with different n_features.")
        n_a, n_b = self.count_, other.count_
        n = n_a + n_b
        mean_a = np.where(n_a > 0, self.mean_, 0.0)
        mean_b = np.where(n_b > 0, other.mean_, 0.0)
        delta = mean_b - mean_a
        safe = np.where(n > 0, n, 1)
        self.mean_ = np.where(n > 0, mean_a + delta * n_b / safe, np.nan)
        self._m2 = self._m2 + other._m2 + delta ** 2 * n_a * n_b / safe
        self.count_ = n
        self.min_ = np.fmin(self.min_, other.min_)
        self.max_ = np.fmax(self.max_, other.max_)
        self.n_chunks_ += other.n_chunks_
        return self

    @property
    def mean(self) -> np.ndarray:
        return self.mean_

    def var(self, ddof: int = 0) -> np.ndarray:
        
        if self.count_ is None:
            return None
        denom = self.count_ - ddof
        out = np.full(self.n_features_, np.nan)
        ok = denom > 0
        out[ok] = np.maximum(self._m2[ok], 0.0) / denom[ok]
        return out

    def std(self, ddof: int = 0) -> np.ndarray:
        v = self.var(ddof)
        return None if v is None else np.sqrt(v)

    def summary(self) -> dict:
        return {"count": self.count_, "mean": self.mean_, "var": self.var(),
                "min": self.min_, "max": self.max_}


class StreamingQuantile:
   

    def __init__(self, capacity: int = 10_000, random_state=None):
        self.capacity = check_positive_int(capacity, "capacity")
        self.random_state = random_state
        self.reset()

    def reset(self) -> "StreamingQuantile":
        self._rng = np.random.default_rng(self.random_state)
        self._res = None
        self._filled = 0
        self.n_seen_ = 0
        return self

    @property
    def is_exact(self) -> bool:
        return self.n_seen_ <= self.capacity

    def update_stats(self, X_chunk) -> "StreamingQuantile":
        X = check_array(X_chunk, name="X_chunk")
        if self._res is None:
            self._res = np.full((self.capacity, X.shape[1]), np.nan)
        elif X.shape[1] != self._res.shape[1]:
            raise ValueError(
                f"X_chunk has {X.shape[1]} features, expected {self._res.shape[1]}."
            )
        free = self.capacity - self._filled
        take = min(free, X.shape[0])
        if take:
            self._res[self._filled:self._filled + take] = X[:take]
            self._filled += take
        rest = X[take:]
        if rest.shape[0]:
            t = self.n_seen_ + take + np.arange(rest.shape[0])
            j = self._rng.integers(0, t + 1)
            keep = j < self.capacity
            self._res[j[keep]] = rest[keep]
        self.n_seen_ += X.shape[0]
        return self

    update = update_stats
    partial_fit = update_stats

    def quantile(self, q):
       
        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError(f"q must be in [0, 1], got {q}.")
        if self._filled == 0:
            raise ValueError("No data seen yet: call update_stats() first.")
        sample = self._res[: self._filled]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  
            return np.nanquantile(sample, q_arr, axis=0)

    def median(self):
        return self.quantile(0.5)


class StreamingHistogram:
   

    def __init__(self, bins: int = 10, range=None, window: int | None = None):
        self.bins = check_positive_int(bins, "bins")
        self.range = range
        self.window = None if window is None else check_positive_int(window, "window")
        self.reset()

    def reset(self) -> "StreamingHistogram":
        self.edges_ = None
        if self.range is not None:
            lo, hi = map(float, self.range)
            if not hi > lo:
                raise ValueError(f"range must satisfy max > min, got {self.range}.")
            self.edges_ = np.linspace(lo, hi, self.bins + 1)
        self._chunks = deque(maxlen=self.window)
        self._total = np.zeros(self.bins, dtype=np.int64)
        self.underflow_ = 0
        self.overflow_ = 0
        self.n_missing_ = 0
        return self

    def update_stats(self, x_chunk) -> "StreamingHistogram":
        x = check_array(x_chunk, name="x_chunk")
        if x.shape[1] != 1:
            raise ValueError(
                f"StreamingHistogram tracks one feature; got {x.shape[1]} columns."
            )
        x = x[:, 0]
        nan = np.isnan(x)
        self.n_missing_ += int(nan.sum())
        x = x[~nan]
        if self.edges_ is None:
            if x.size == 0:
                return self          
            lo, hi = float(x.min()), float(x.max())
            if hi == lo:             
                lo, hi = lo - 0.5, hi + 0.5
            self.edges_ = np.linspace(lo, hi, self.bins + 1)
        lo, hi = self.edges_[0], self.edges_[-1]
        self.underflow_ += int((x < lo).sum())
        self.overflow_ += int((x > hi).sum())
        counts, _ = np.histogram(x[(x >= lo) & (x <= hi)], bins=self.edges_)
        self._chunks.append(counts)
        self._total += counts
        return self

    update = update_stats
    partial_fit = update_stats

    @property
    def counts(self) -> np.ndarray:
        if self.window is None:
            return self._total.copy()
        if not self._chunks:
            return np.zeros(self.bins, dtype=np.int64)
        return np.sum(np.stack(self._chunks), axis=0)

    def density(self) -> np.ndarray:
        c = self.counts.astype(float)
        total = c.sum()
        if total == 0:
            return np.zeros_like(c)
        return c / (total * np.diff(self.edges_))



def update_stats(state: StreamingStats | None, X_chunk) -> StreamingStats:
    
    if state is None:
        state = StreamingStats()
    return state.update_stats(X_chunk)


def chunked_mean(chunks: Iterable) -> np.ndarray:
    s = StreamingStats()
    for c in chunks:
        s.update_stats(c)
    return s.mean


def chunked_variance(chunks: Iterable, ddof: int = 0) -> np.ndarray:
    s = StreamingStats()
    for c in chunks:
        s.update_stats(c)
    return s.var(ddof)


def chunked_quantile(chunks: Iterable, q, capacity: int = 10_000,
                     random_state=None) -> np.ndarray:
    s = StreamingQuantile(capacity, random_state)
    for c in chunks:
        s.update_stats(c)
    return s.quantile(q)


def chunked_histogram(chunks: Iterable, bins: int = 10, range=None):
    h = StreamingHistogram(bins, range)
    for c in chunks:
        h.update_stats(c)
    return h.counts, h.edges_
