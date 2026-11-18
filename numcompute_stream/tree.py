

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from ._validation import (NotFittedError, check_array, check_is_fitted,
                          check_positive_int, check_X_y)

__all__ = ["DecisionTreeClassifier", "best_split", "gini", "entropy", "SplitResult"]

_GAIN_TOL = 1e-12




def gini(counts: np.ndarray) -> np.ndarray:
    
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1)
    safe = np.where(n > 0, n, 1.0)
    p = counts / safe[..., None]
    return np.where(n > 0, 1.0 - (p * p).sum(axis=-1), 0.0)


def entropy(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1)
    safe = np.where(n > 0, n, 1.0)
    p = counts / safe[..., None]
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return np.where(n > 0, -(p * logp).sum(axis=-1), 0.0)


_CRITERIA = {"gini": gini, "entropy": entropy}


class SplitResult(NamedTuple):

    feature: int          
    threshold: float      
    gain: float           
    missing_left: bool    
    n_left: int
    n_right: int


def best_split(X: np.ndarray, y_idx: np.ndarray, n_classes: int,
               criterion: str = "gini", features=None) -> SplitResult | None:
    
    imp = _CRITERIA[criterion]
    n = X.shape[0]
    feats = np.arange(X.shape[1]) if features is None else np.asarray(features, dtype=int)
    if n < 2 or feats.size == 0:
        return None
    Xs = X[:, feats]
    order = np.argsort(Xs, axis=0, kind="stable")         
    xs = np.take_along_axis(Xs, order, axis=0)              
    Y = np.eye(n_classes)[y_idx]                             
    Ys = Y[order]                                            
    cum = np.cumsum(Ys, axis=0)                             
    nan_sorted = np.isnan(xs)
    nan_counts = (Ys * nan_sorted[..., None]).sum(axis=0)    
    observed = cum[-1] - nan_counts                         

    left = cum[:-1]                                          
    right = observed[None] - left
    with np.errstate(invalid="ignore"):
        valid = xs[1:] > xs[:-1]                             
    if not valid.any():
        return None

    parent = imp(Y.sum(axis=0))
    nL, nR = left.sum(-1), right.sum(-1)                     
    n_nan = nan_counts.sum(-1)                               
    nLa, nRb = nL + n_nan, nR + n_nan
    after_a = (nLa * imp(left + nan_counts) + nR * imp(right)) / n
    after_b = (nL * imp(left) + nRb * imp(right + nan_counts)) / n
    gain_a, gain_b = parent - after_a, parent - after_b
    go_left = gain_a >= gain_b
    gain = np.where(go_left, gain_a, gain_b)
    gain = np.where(valid, gain, -np.inf)

    best = gain.max()
    if not best > _GAIN_TOL:
        return None
    cand = (gain >= best - _GAIN_TOL).T
    fi, pos = np.unravel_index(np.argmax(cand), cand.shape)
    lo, hi = xs[pos, fi], xs[pos + 1, fi]
    thr = lo + (hi - lo) / 2.0
    if not thr < hi:          
        thr = lo
    has_nan = n_nan[fi] > 0
    ml = bool(go_left[pos, fi]) if has_nan else bool(nL[pos, fi] >= nR[pos, fi])
    n_left = int(nLa[pos, fi] if (has_nan and ml) else nL[pos, fi])
    return SplitResult(int(feats[fi]), float(thr), float(gain[pos, fi]), ml,
                       n_left, n - n_left)


class DecisionTreeClassifier:
   

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_features=None,
                 criterion: str = "gini", buffer_size: int = 500,
                 min_impurity_decrease: float = 0.0, random_state=None):
        self.max_depth = check_positive_int(max_depth, "max_depth", minimum=0)
        self.min_samples_split = check_positive_int(min_samples_split, "min_samples_split", 2)
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}.")
        self.criterion = criterion
        self.buffer_size = check_positive_int(buffer_size, "buffer_size")
        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be >= 0.")
        self.min_impurity_decrease = float(min_impurity_decrease)
        self._check_max_features(max_features)
        self.max_features = max_features
        self.random_state = random_state
        self.reset()

    @staticmethod
    def _check_max_features(mf):
        if mf is None or mf in ("sqrt", "log2"):
            return
        if isinstance(mf, bool):
            raise TypeError("max_features must be None, int, float, 'sqrt' or 'log2'.")
        if isinstance(mf, (int, np.integer)):
            if mf < 1:
                raise ValueError(f"max_features as an int must be >= 1, got {mf}.")
            return
        if isinstance(mf, float):
            if not 0.0 < mf <= 1.0:
                raise ValueError(f"max_features as a float must be in (0, 1], got {mf}.")
            return
        raise ValueError(f"max_features must be None, int, float, 'sqrt' or 'log2', got {mf!r}.")

    def _n_features_to_try(self, d: int) -> int:
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d))) if d > 1 else 1
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(int(mf), d)

    def reset(self):
        self.classes_ = None
        self._fit_classes = None
        self.n_features_in_ = None
        self._width = None              
        self._buf_X, self._buf_y, self._buf_n = [], [], 0
        self.n_retrains_ = 0
        self.retrain_times_ = []
        self.samples_seen_ = 0
        self.last_retrain_at_ = None    
        self.feature_ = None
        return self

    def fit(self, X, y):
        
        X, y = check_X_y(X, y)
        self.reset()
        self._update_classes(y)
        self._width = X.shape[1]
        self.samples_seen_ = X.shape[0]
        self._retrain(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
       
        X, y = check_X_y(X_chunk, y_chunk)
        if classes is not None:
            self._update_classes(np.asarray(classes))
        self._update_classes(y)
        d = X.shape[1]
        if self._width is not None and d < self._width:
            raise ValueError(
                f"X_chunk has {d} features but earlier chunks had {self._width}; the "
                f"feature count may grow (appended columns) but never shrink."
            )
        if self._width is not None and d > self._width:
            self._buf_X = [np.pad(b, ((0, 0), (0, d - b.shape[1]))) for b in self._buf_X]
        self._width = d
        self._buf_X.append(X)
        self._buf_y.append(y)
        self._buf_n += X.shape[0]
        self.samples_seen_ += X.shape[0]
        if self._buf_n >= self.buffer_size:
            self.flush()
        return self

    def flush(self):
        
        if self._buf_n == 0:
            return self
        widths = {b.shape[1] for b in self._buf_X}
        if len(widths) > 1:          
            w = max(widths)
            self._buf_X = [np.pad(b, ((0, 0), (0, w - b.shape[1]))) for b in self._buf_X]
        X = np.concatenate(self._buf_X)
        y = np.concatenate(self._buf_y)
        self._buf_X, self._buf_y, self._buf_n = [], [], 0
        self._retrain(X, y)
        return self

    @property
    def buffer_fill(self) -> int:
        return self._buf_n

    def _update_classes(self, y):
        y = np.asarray(y)
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            return
        if self.classes_.dtype.kind != y.dtype.kind and \
                ("U" in (self.classes_.dtype.kind, y.dtype.kind)):
            self.classes_ = self.classes_.astype(str)
            y = y.astype(str)
        self.classes_ = np.union1d(self.classes_, y)

    def _encode(self, y, classes):
        if classes.dtype.kind in "US":
            y = y.astype(str)
        return np.searchsorted(classes, y)

    def _retrain(self, X, y):
        t0 = time.perf_counter()
        self._fit_classes = self.classes_.copy()
        self._grow(X, self._encode(y, self._fit_classes), len(self._fit_classes))
        self._trained_with_nan = bool(np.isnan(X).any())
        self.n_features_in_ = X.shape[1]
        self.retrain_times_.append(time.perf_counter() - t0)
        self.n_retrains_ += 1
        self.last_retrain_at_ = self.samples_seen_

    def _grow(self, X, y_idx, k):
        imp = _CRITERIA[self.criterion]
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        m = self._n_features_to_try(d)
        feat, thr, left, right, miss, value, impur, nsamp, depth = ([] for _ in range(9))
        importances = np.zeros(d)

        def new_node(rows, dep):
            counts = np.bincount(y_idx[rows], minlength=k).astype(float)
            feat.append(-1); thr.append(np.nan); left.append(-1); right.append(-1)
            miss.append(True); value.append(counts); impur.append(float(imp(counts)))
            nsamp.append(rows.size); depth.append(dep)
            return len(feat) - 1

        root = new_node(np.arange(X.shape[0]), 0)
        stack = [(root, np.arange(X.shape[0]))]
        while stack:
            node, rows = stack.pop()
            counts = value[node]
            if (depth[node] >= self.max_depth or rows.size < self.min_samples_split
                    or np.count_nonzero(counts) <= 1):
                continue
            features = None
            if m < d:
                features = np.sort(rng.choice(d, size=m, replace=False))
            s = best_split(X[rows], y_idx[rows], k, self.criterion, features)
            if s is None or s.gain <= self.min_impurity_decrease + _GAIN_TOL:
                continue
            col = X[rows, s.feature]
            go = np.where(np.isnan(col), s.missing_left, col <= s.threshold)
            lrows, rrows = rows[go], rows[~go]
            feat[node], thr[node], miss[node] = s.feature, s.threshold, s.missing_left
            importances[s.feature] += rows.size * s.gain
            l_id = new_node(lrows, depth[node] + 1)
            r_id = new_node(rrows, depth[node] + 1)
            left[node], right[node] = l_id, r_id
            stack.append((r_id, rrows))      
            stack.append((l_id, lrows))

        self.feature_ = np.array(feat, dtype=np.int64)
        self.threshold_ = np.array(thr, dtype=float)
        self.left_ = np.array(left, dtype=np.int64)
        self.right_ = np.array(right, dtype=np.int64)
        self.missing_left_ = np.array(miss, dtype=bool)
        self.value_ = np.vstack(value)
        self.impurity_ = np.array(impur)
        self.n_node_samples_ = np.array(nsamp, dtype=np.int64)
        self.node_depth_ = np.array(depth, dtype=np.int64)
        tot = importances.sum()
        self.feature_importances_ = importances / tot if tot > 0 else importances

    def _check_predict_X(self, X):
        check_is_fitted(self, "feature_", "predict()")
        X = check_array(X)
        if X.shape[1] < self.n_features_in_ or X.shape[1] > max(self._width, self.n_features_in_):
            raise ValueError(
                f"X has {X.shape[1]} features, but the tree was trained on "
                f"{self.n_features_in_}" + (f" (and has seen up to {self._width})"
                                            if self._width != self.n_features_in_ else "") + "."
            )
        return X

    def apply(self, X):
        
        X = self._check_predict_X(X)
        node = np.zeros(X.shape[0], dtype=np.int64)
        rows = np.arange(X.shape[0])
        for _ in range(self.max_depth + 1):
            f = self.feature_[node]
            internal = f >= 0
            if not internal.any():
                break
            x = X[rows, np.maximum(f, 0)]
            go_left = np.where(np.isnan(x), self.missing_left_[node], x <= self.threshold_[node])
            nxt = np.where(go_left, self.left_[node], self.right_[node])
            node = np.where(internal, nxt, node)
        return node

    def predict_proba(self, X):
       
        leaf = self.apply(X)
        counts = self.value_[leaf]
        p = counts / counts.sum(axis=1, keepdims=True)
        if len(self._fit_classes) == len(self.classes_):
            return p
        out = np.zeros((p.shape[0], len(self.classes_)))
        out[:, np.searchsorted(self.classes_, self._fit_classes)] = p
        return out

    def predict(self, X):
        
        leaf = self.apply(X)
        return self._fit_classes[np.argmax(self.value_[leaf], axis=1)]

    def score(self, X, y):
        X, y = check_X_y(X, y)
        pred = self.predict(X)
        if pred.dtype.kind in "US" or y.dtype.kind in "US":
            pred, y = pred.astype(str), y.astype(str)
        return float(np.mean(pred == y))

    @property
    def is_fitted(self) -> bool:
        return self.feature_ is not None

    def get_depth(self) -> int:
        check_is_fitted(self, "feature_", "get_depth()")
        return int(self.node_depth_.max())

    def get_n_leaves(self) -> int:
        check_is_fitted(self, "feature_", "get_n_leaves()")
        return int((self.feature_ < 0).sum())

    def memory_footprint(self) -> int:
        b = sum(a.nbytes for a in self._buf_X) + sum(a.nbytes for a in self._buf_y)
        if self.feature_ is not None:
            b += sum(a.nbytes for a in (self.feature_, self.threshold_, self.left_, self.right_,
                                        self.missing_left_, self.value_, self.impurity_,
                                        self.n_node_samples_, self.node_depth_))
        return int(b)

    def export_text(self, feature_names=None, decimals: int = 3) -> str:
       
        check_is_fitted(self, "feature_", "export_text()")
        names = feature_names or [f"x{j}" for j in range(self.n_features_in_)]
        lines = []

        def walk(node, indent):
            pad = "|   " * indent + "|--- "
            if self.feature_[node] < 0:
                cls = self._fit_classes[np.argmax(self.value_[node])]
                lines.append(f"{pad}class: {cls}  (n={self.n_node_samples_[node]})")
                return
            f, t = names[self.feature_[node]], round(self.threshold_[node], decimals)
            tag = " [NaN]" if getattr(self, "_trained_with_nan", False) else ""
            nan_l = tag if self.missing_left_[node] else ""
            nan_r = "" if self.missing_left_[node] else tag
            lines.append(f"{pad}{f} <= {t}{nan_l}")
            walk(self.left_[node], indent + 1)
            lines.append(f"{pad}{f} >  {t}{nan_r}")
            walk(self.right_[node], indent + 1)

        walk(0, 0)
        return "\n".join(lines)

    def __repr__(self):
        return (f"DecisionTreeClassifier(max_depth={self.max_depth}, criterion="
                f"{self.criterion!r}, buffer_size={self.buffer_size})")


DecisionTree = DecisionTreeClassifier
__all__.append("DecisionTree")
