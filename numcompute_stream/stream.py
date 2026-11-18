

from __future__ import annotations

import logging
import time
import tracemalloc
from typing import Iterable, Iterator

import numpy as np

from ._validation import check_positive_int, check_y
from .io import write_csv
from .metrics import Accuracy, F1Score, ROCAUC, RollingMetric

__all__ = ["StreamTrainer", "iter_chunks"]

log = logging.getLogger("numcompute_stream")


def iter_chunks(X, y=None, chunk_size: int = 500) -> Iterator:
   
    chunk_size = check_positive_int(chunk_size, "chunk_size")
    n = len(X)
    if y is not None and len(y) != n:
        raise ValueError(f"X has {n} rows but y has {len(y)}.")
    for s in range(0, n, chunk_size):
        yield (X[s:s + chunk_size], y[s:s + chunk_size]) if y is not None else X[s:s + chunk_size]


def _final(model):
    return getattr(model, "final_estimator", model)


class StreamTrainer:
    

    def __init__(self, model, metrics: dict | None = None, rolling_window: int | None = None,
                 track_auc: bool = True, trace_memory: bool = False, name: str | None = None,
                 verbose: bool = False):
        if not (callable(getattr(model, "partial_fit", None))
                and callable(getattr(model, "predict", None))):
            raise TypeError("model must implement partial_fit() and predict().")
        self.model = model
        self.metric_factories = metrics or {
            "accuracy": Accuracy,
            "f1_macro": lambda: F1Score(average="macro"),
        }
        self.rolling_window = rolling_window
        self.track_auc = track_auc
        self.trace_memory = trace_memory
        self.name = name or type(_final(model)).__name__
        self.verbose = verbose
        self.reset_log()

    def reset_log(self):
        self.cumulative = {k: f() for k, f in self.metric_factories.items()}
        self.rolling = RollingMetric(Accuracy, self.rolling_window) if self.rolling_window else None
        self._auc = None
        self.records = []
        self.n_chunks_ = 0
        self.last_y_true_ = self.last_y_pred_ = None
        return self

    def score_chunk(self, X, y) -> dict:
       
        y = check_y(y)
        rec = {"chunk_accuracy": np.nan, "predict_time": np.nan}
        fitted = bool(getattr(_final(self.model), "is_fitted", True))
        if fitted:
            t0 = time.perf_counter()
            pred = np.asarray(self.model.predict(X))
            rec["predict_time"] = time.perf_counter() - t0
            yc, pc = (y.astype(str), pred.astype(str)) if "U" in (y.dtype.kind, pred.dtype.kind) \
                else (y, pred)
            rec["chunk_accuracy"] = float(np.mean(yc == pc)) if len(y) else np.nan
            for m in self.cumulative.values():
                m.update(y, pred)
            if self.rolling is not None:
                self.rolling.update(y, pred)
            self._update_auc(X, y)
            self.last_y_true_, self.last_y_pred_ = y, pred
        for k, m in self.cumulative.items():
            rec[f"cumulative_{k}"] = m.result()
        if self.rolling is not None:
            rec["rolling_accuracy"] = self.rolling.result()
        if self.track_auc:
            rec["roc_auc"] = self._auc.result() if self._auc is not None else np.nan
        return rec

    def _update_auc(self, X, y):
        if not self.track_auc or not callable(getattr(self.model, "predict_proba", None)):
            return
        classes = getattr(_final(self.model), "classes_", None)
        if classes is None or len(classes) != 2:
            return
        if self._auc is None:
            self._auc = ROCAUC(pos_label=classes[1])
        proba = self.model.predict_proba(X)
        self._auc.update(y, proba[:, 1])

    def fit_chunk(self, X, y) -> dict:
        
        est = _final(self.model)
        before = getattr(est, "n_retrains_", 0)
        if self.trace_memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        self.model.partial_fit(X, y)
        rec = {"fit_time": time.perf_counter() - t0}
        if self.trace_memory:
            rec["peak_alloc_bytes"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        after = getattr(est, "n_retrains_", 0)
        rec["retrained"] = after > before
        rec["n_retrains"] = after
        rec["buffer_fill"] = getattr(est, "buffer_fill", np.nan)
        mf = getattr(self.model, "memory_footprint", None)
        rec["memory_bytes"] = mf() if callable(mf) else np.nan
        return rec

    def step(self, X, y) -> dict:
        rec = {"chunk": self.n_chunks_, "n_samples": len(y)}
        rec.update(self.score_chunk(X, y))
        rec.update(self.fit_chunk(X, y))
        self.records.append(rec)
        self.n_chunks_ += 1
        self._log(rec)
        return rec

    def run(self, chunks: Iterable, max_chunks: int | None = None, callback=None):
       
        for i, (X, y) in enumerate(chunks):
            if max_chunks is not None and i >= max_chunks:
                break
            rec = self.step(X, y)
            if callback is not None:
                callback(self, rec)
        return self

    def _log(self, r):
        acc = r["chunk_accuracy"]
        cum = r.get("cumulative_accuracy", np.nan)
        msg = (f"[{self.name}] chunk {r['chunk']:>3} | n={r['n_samples']:>4} | "
               f"acc={acc:6.3f} | cum_acc={cum:6.3f} | "
               f"{'RETRAIN' if r['retrained'] else 'buffer ':>7} "
               f"fill={r['buffer_fill']:>5} | fit={r['fit_time'] * 1e3:7.2f} ms | "
               f"mem={r['memory_bytes'] / 1024:8.1f} KiB")
        log.info(msg)
        if self.verbose:
            print(msg)

    @property
    def history(self) -> dict:
        if not self.records:
            return {}
        keys = list(self.records[-1])
        for r in self.records:
            for k in r:
                if k not in keys:
                    keys.append(k)
        return {k: np.array([r.get(k, np.nan) for r in self.records]) for k in keys}

    def retrain_chunks(self) -> np.ndarray:
        h = self.history
        return h["chunk"][h["retrained"].astype(bool)] if h else np.array([], dtype=int)

    def to_csv(self, path) -> None:
        if not self.records:
            raise ValueError("Nothing logged yet.")
        write_csv(path, self.history)

    def summary(self) -> dict:
        h = self.history
        if not h:
            return {}
        out = {f"final_{k}": m.result() for k, m in self.cumulative.items()}
        out.update({
            "chunks": int(len(h["chunk"])),
            "samples": int(h["n_samples"].sum()),
            "retrains": int(h["n_retrains"][-1]),
            "total_fit_time_s": float(np.nansum(h["fit_time"])),
            "peak_memory_bytes": float(np.nanmax(h["memory_bytes"])),
        })
        if self.rolling is not None:
            out["final_rolling_accuracy"] = self.rolling.result()
        return out
