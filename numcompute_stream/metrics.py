

from __future__ import annotations

import numpy as np

from ._validation import check_positive_int, check_y

__all__ = [
    "ConfusionMatrix", "Accuracy", "Precision", "Recall", "F1Score", "ROCAUC",
    "RollingMetric",
    "confusion_matrix", "accuracy", "precision", "recall", "f1_score", "roc_auc_score",
]


def _pair(y_true, y_pred, second="y_pred"):
    yt = check_y(y_true, name="y_true")
    yp = check_y(y_pred, name=second)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and {second} have different lengths: {yt.shape[0]} vs {yp.shape[0]}."
        )
    if yt.dtype.kind in "US" or yp.dtype.kind in "US":
        yt, yp = yt.astype(str), yp.astype(str)
    return yt, yp


class _StreamingMetric:

    name = "metric"

    def __repr__(self):
        return f"{type(self).__name__}()"


class ConfusionMatrix(_StreamingMetric):
   

    name = "confusion_matrix"

    def __init__(self, labels=None):
        self._fixed = None if labels is None else np.unique(np.asarray(labels))
        self.reset()

    def reset(self):
        self.labels_ = self._fixed
        k = 0 if self._fixed is None else len(self._fixed)
        self.matrix_ = np.zeros((k, k), dtype=np.int64)
        self.n_samples_ = 0
        return self

    def _grow(self, new_labels):
        if self.labels_ is None or len(self.labels_) == 0:
            self.labels_ = new_labels
            self.matrix_ = np.zeros((len(new_labels),) * 2, dtype=np.int64)
            return
        if self.labels_.dtype.kind != new_labels.dtype.kind:
            self.labels_ = self.labels_.astype(str)
            new_labels = new_labels.astype(str)
        merged = np.union1d(self.labels_, new_labels)
        if len(merged) == len(self.labels_):
            return
        pos = np.searchsorted(merged, self.labels_)
        m = np.zeros((len(merged),) * 2, dtype=np.int64)
        m[np.ix_(pos, pos)] = self.matrix_
        self.labels_, self.matrix_ = merged, m

    def update(self, y_true, y_pred):
        yt, yp = _pair(y_true, y_pred)
        if yt.size == 0:
            return self
        seen = np.union1d(yt, yp)
        if self._fixed is not None:
            extra = np.setdiff1d(seen, self._fixed)
            if extra.size:
                raise ValueError(f"Labels {extra.tolist()} are not in the fixed label set.")
        else:
            self._grow(seen)
        if self.labels_.dtype.kind in "US":
            yt, yp = yt.astype(str), yp.astype(str)
        k = len(self.labels_)
        i = np.searchsorted(self.labels_, yt)
        j = np.searchsorted(self.labels_, yp)
        self.matrix_ += np.bincount(i * k + j, minlength=k * k).reshape(k, k)
        self.n_samples_ += yt.size
        return self

    def result(self):
        return self.matrix_.copy()


class _FromConfusion(_StreamingMetric):

    def __init__(self, average: str = "macro", pos_label=1, zero_division: float = 0.0):
        if average not in ("binary", "macro", "micro", "weighted"):
            raise ValueError("average must be 'binary', 'macro', 'micro' or 'weighted'.")
        self.average = average
        self.pos_label = pos_label
        self.zero_division = float(zero_division)
        self.cm = ConfusionMatrix()

    def reset(self):
        self.cm.reset()
        return self

    def update(self, y_true, y_pred):
        self.cm.update(y_true, y_pred)
        return self

    def _parts(self):
        m = self.cm.matrix_.astype(float)
        tp = np.diag(m)
        return tp, m.sum(axis=0), m.sum(axis=1)   

    def _ratio(self, num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        out = np.full(np.broadcast(num, den).shape, self.zero_division)
        np.divide(num, den, out=out, where=den > 0)
        return out

    def _reduce(self, per_label, num, den, support):
        if self.cm.n_samples_ == 0:
            return float("nan")
        if self.average == "micro":
            return float(self._ratio(num.sum(), den.sum()))
        if self.average == "binary":
            labels = self.cm.labels_
            hit = np.flatnonzero(labels.astype(str) == str(self.pos_label))
            return float(per_label[hit[0]]) if hit.size else self.zero_division
        if self.average == "weighted":
            w = support.sum()
            return float((per_label * support).sum() / w) if w > 0 else self.zero_division
        return float(per_label.mean())

    def per_label(self):
        return self._per_label()


class Accuracy(_FromConfusion):

    name = "accuracy"

    def __init__(self):
        super().__init__()

    def result(self):
        n = self.cm.n_samples_
        return float("nan") if n == 0 else float(np.trace(self.cm.matrix_) / n)


class Precision(_FromConfusion):

    name = "precision"

    def _per_label(self):
        tp, pred, _ = self._parts()
        return self._ratio(tp, pred)

    def result(self):
        tp, pred, actual = self._parts()
        return self._reduce(self._ratio(tp, pred), tp, pred, actual)


class Recall(_FromConfusion):

    name = "recall"

    def _per_label(self):
        tp, _, actual = self._parts()
        return self._ratio(tp, actual)

    def result(self):
        tp, _, actual = self._parts()
        return self._reduce(self._ratio(tp, actual), tp, actual, actual)


class F1Score(_FromConfusion):

    name = "f1"

    def _per_label(self):
        tp, pred, actual = self._parts()
        return self._ratio(2 * tp, pred + actual)

    def result(self):
        tp, pred, actual = self._parts()
        return self._reduce(self._ratio(2 * tp, pred + actual), 2 * tp, pred + actual, actual)


class ROCAUC(_StreamingMetric):
    

    name = "roc_auc"

    def __init__(self, pos_label=1, n_bins: int = 1000, exact: bool = False):
        self.pos_label = pos_label
        self.n_bins = check_positive_int(n_bins, "n_bins")
        self.exact = exact
        self.reset()

    def reset(self):
        self._pos = np.zeros(self.n_bins, dtype=np.int64)
        self._neg = np.zeros(self.n_bins, dtype=np.int64)
        self._scores, self._truth = [], []
        return self

    def update(self, y_true, y_score):
        yt = check_y(y_true, name="y_true")
        s = np.asarray(y_score, dtype=float).ravel()
        if s.shape[0] != yt.shape[0]:
            raise ValueError(
                f"y_true and y_score have different lengths: {yt.shape[0]} vs {s.shape[0]}."
            )
        if np.isnan(s).any():
            raise ValueError("y_score contains NaN.")
        is_pos = yt.astype(str) == str(self.pos_label)
        if self.exact:
            self._scores.append(s)
            self._truth.append(is_pos)
            return self
        if s.size and (s.min() < 0 or s.max() > 1):
            raise ValueError("Binned ROCAUC expects scores in [0, 1] (probabilities); "
                             "use exact=True for arbitrary scores.")
        b = np.minimum((s * self.n_bins).astype(int), self.n_bins - 1)
        self._pos += np.bincount(b[is_pos], minlength=self.n_bins)
        self._neg += np.bincount(b[~is_pos], minlength=self.n_bins)
        return self

    def result(self):
        if self.exact:
            if not self._scores:
                return float("nan")
            return _rank_auc(np.concatenate(self._truth), np.concatenate(self._scores))
        P, N = self._pos.sum(), self._neg.sum()
        if P == 0 or N == 0:
            return float("nan")
        neg_below = np.concatenate(([0], np.cumsum(self._neg)[:-1]))
        return float((self._pos * (neg_below + 0.5 * self._neg)).sum() / (P * N))


def _rank_auc(is_pos, scores):
    P, N = is_pos.sum(), (~is_pos).sum()
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[is_pos].sum() - P * (P + 1) / 2) / (P * N))


class RollingMetric(_StreamingMetric):
   

    def __init__(self, metric=Accuracy, window: int = 1000):
        self.metric = metric
        self.window = check_positive_int(window, "window")
        self.name = f"rolling_{getattr(metric(), 'name', 'metric')}"
        self.reset()

    def reset(self):
        self._yt = None
        self._yp = None
        return self

    def update(self, y_true, y_pred):
        yt = check_y(y_true, name="y_true")
        yp = np.asarray(y_pred).ravel()
        if yp.shape[0] != yt.shape[0]:
            raise ValueError("y_true and y_pred have different lengths.")
        if self._yt is None:
            self._yt, self._yp = yt[-self.window:], yp[-self.window:]
        else:
            self._yt = np.concatenate([self._yt, yt])[-self.window:]
            self._yp = np.concatenate([self._yp, yp])[-self.window:]
        return self

    def result(self):
        if self._yt is None or self._yt.size == 0:
            return float("nan")
        return self.metric().update(self._yt, self._yp).result()



def confusion_matrix(y_true, y_pred, labels=None):
    cm = ConfusionMatrix(labels).update(y_true, y_pred)
    return cm.result(), cm.labels_


def accuracy(y_true, y_pred):
    return Accuracy().update(y_true, y_pred).result()


def precision(y_true, y_pred, average="macro", pos_label=1, zero_division=0.0):
    return Precision(average, pos_label, zero_division).update(y_true, y_pred).result()


def recall(y_true, y_pred, average="macro", pos_label=1, zero_division=0.0):
    return Recall(average, pos_label, zero_division).update(y_true, y_pred).result()


def f1_score(y_true, y_pred, average="macro", pos_label=1, zero_division=0.0):
    return F1Score(average, pos_label, zero_division).update(y_true, y_pred).result()


def roc_auc_score(y_true, y_score, pos_label=1):
    return ROCAUC(pos_label=pos_label, exact=True).update(y_true, y_score).result()
