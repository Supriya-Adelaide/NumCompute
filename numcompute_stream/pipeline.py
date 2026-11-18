
from __future__ import annotations

import numpy as np

from .io import Frame

__all__ = ["Pipeline", "ColumnTransformer", "make_pipeline"]


def _has(obj, *methods):
    return all(callable(getattr(obj, m, None)) for m in methods)


class Pipeline:
    

    def __init__(self, steps):
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ValueError("steps must be a non-empty list of (name, step) tuples.")
        names = []
        for i, item in enumerate(steps):
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
                raise TypeError(f"Step {i} must be a (name, step) tuple, got {item!r}.")
            names.append(item[0])
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique, got {names}.")
        for name, step in steps[:-1]:
            if not _has(step, "partial_fit", "transform"):
                raise TypeError(
                    f"Intermediate step {name!r} ({type(step).__name__}) must implement "
                    f"partial_fit() and transform()."
                )
        last_name, last = steps[-1]
        if not _has(last, "partial_fit"):
            raise TypeError(f"Final step {last_name!r} must implement partial_fit().")
        self.steps = list(steps)

    @property
    def named_steps(self) -> dict:
        return dict(self.steps)

    def __getitem__(self, key):
        return self.named_steps[key] if isinstance(key, str) else self.steps[key][1]

    def __len__(self):
        return len(self.steps)

    @property
    def final_estimator(self):
        return self.steps[-1][1]

    def __repr__(self):
        inner = ", ".join(f"({n!r}, {s!r})" for n, s in self.steps)
        return f"Pipeline([{inner}])"

    def _transformers(self):
        return [s for _, s in self.steps[:-1]]

    def partial_fit(self, X, y=None):
       
        Xt = X
        for t in self._transformers():
            Xt = t.partial_fit(Xt, y).transform(Xt)
        self.final_estimator.partial_fit(Xt, y)
        return self

    def fit(self, X, y=None):
        Xt = X
        for t in self._transformers():
            Xt = t.fit(Xt, y).transform(Xt)
        self.final_estimator.fit(Xt, y)
        return self

    def flush(self):
        if _has(self.final_estimator, "flush"):
            self.final_estimator.flush()
        return self

    def _pre(self, X):
        Xt = X
        for t in self._transformers():
            Xt = t.transform(Xt)
        return Xt

    def transform(self, X):
        Xt = self._pre(X)
        if _has(self.final_estimator, "transform"):
            Xt = self.final_estimator.transform(Xt)
        return Xt

    def predict(self, X):
        return self.final_estimator.predict(self._pre(X))

    def predict_proba(self, X):
        return self.final_estimator.predict_proba(self._pre(X))

    def score(self, X, y):
        return self.final_estimator.score(self._pre(X), y)

    @property
    def is_fitted(self) -> bool:
        return bool(getattr(self.final_estimator, "is_fitted", False))

    def memory_footprint(self) -> int:
        return int(sum(s.memory_footprint() for _, s in self.steps
                       if _has(s, "memory_footprint")))


def make_pipeline(*steps) -> Pipeline:
    return Pipeline([(type(s).__name__.lower(), s) for s in steps])


class ColumnTransformer:
   

    def __init__(self, transformers):
        if not transformers:
            raise ValueError("transformers must be a non-empty list.")
        names = [t[0] for t in transformers]
        if len(set(names)) != len(names):
            raise ValueError(f"Transformer names must be unique, got {names}.")
        for name, t, _ in transformers:
            if t != "passthrough" and not _has(t, "partial_fit", "transform"):
                raise TypeError(f"Transformer {name!r} must implement partial_fit() and transform().")
        self.transformers = list(transformers)
        self._widths = None

    def _select(self, X, cols):
        if isinstance(X, Frame):
            if not all(isinstance(c, str) for c in cols):
                raise TypeError("Use column names to select from a Frame.")
            return X.to_array(cols)
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if not all(isinstance(c, (int, np.integer)) for c in cols):
            raise TypeError("Use integer column indices to select from an array.")
        return X[:, list(cols)]

    def partial_fit(self, X, y=None):
        for _, t, cols in self.transformers:
            if t != "passthrough":
                t.partial_fit(self._select(X, cols), y)
        return self

    def fit(self, X, y=None):
        for _, t, cols in self.transformers:
            if t != "passthrough":
                t.fit(self._select(X, cols), y)
        return self

    def transform(self, X):
        blocks = []
        for _, t, cols in self.transformers:
            sub = self._select(X, cols)
            blocks.append(np.asarray(sub, dtype=float) if t == "passthrough" else t.transform(sub))
        widths = [b.shape[1] for b in blocks]
        if self._widths is not None and any(w != w0 for w, w0 in zip(widths[:-1], self._widths[:-1])):
            raise RuntimeError(
                "A non-final block changed width between chunks; put growing "
                "transformers (e.g. OneHotEncoder) last."
            )
        self._widths = widths
        return np.hstack(blocks)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> list:
        names = []
        for name, t, cols in self.transformers:
            if t == "passthrough" or not _has(t, "get_feature_names_out"):
                names += [str(c) for c in cols]
            else:
                names += t.get_feature_names_out([str(c) for c in cols])
        return names
