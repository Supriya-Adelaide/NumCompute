
from __future__ import annotations

import numpy as np

__all__ = [
    "NotFittedError",
    "check_array",
    "check_y",
    "check_X_y",
    "check_is_fitted",
    "check_positive_int",
]


class NotFittedError(RuntimeError):
    """Raised when a method needs state that has not been learned yet."""


def check_array(X, *, name: str = "X", allow_nan: bool = True,
                ensure_min_samples: int = 1, dtype=float) -> np.ndarray:
    
    if X is None:
        raise TypeError(f"{name} must be array-like, got None.")
    try:
        arr = np.asarray(X, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must contain only numeric values; could not convert to "
            f"{np.dtype(dtype).name} ({exc})."
        ) from None
    if arr.ndim == 0:
        raise ValueError(f"{name} must be 1-D or 2-D, got a scalar.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ValueError(
            f"{name} must be 2-D with shape (n_samples, n_features), got "
            f"{arr.ndim}-D array with shape {arr.shape}."
        )
    if arr.shape[0] < ensure_min_samples:
        raise ValueError(
            f"{name} needs at least {ensure_min_samples} sample(s), got shape "
            f"{arr.shape}."
        )
    if np.issubdtype(arr.dtype, np.floating):
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains infinite values.")
        if not allow_nan and np.isnan(arr).any():
            raise ValueError(
                f"{name} contains NaN values; impute them first (see "
                f"numcompute_stream.preprocessing.SimpleImputer)."
            )
    return arr


def check_y(y, *, name: str = "y") -> np.ndarray:
   
    if y is None:
        raise TypeError(f"{name} must be array-like, got None.")
    arr = np.asarray(y)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D with shape (n_samples,), got shape {arr.shape}."
        )
    if arr.dtype.kind == "f":
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN labels; drop those rows first.")
        if np.all(arr == np.round(arr)):
            arr = arr.astype(np.int64)
    if arr.dtype.kind == "O":
        if any(v is None for v in arr):
            raise ValueError(f"{name} contains missing (None) labels.")
        arr = arr.astype(str)
    return arr


def check_X_y(X, y, *, allow_nan: bool = True):
    
    X = check_array(X, allow_nan=allow_nan)
    y = check_y(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent numbers of samples: X has "
            f"{X.shape[0]} rows but y has {y.shape[0]} labels."
        )
    return X, y


def check_is_fitted(obj, attribute: str, what: str = "this method") -> None:
    if getattr(obj, attribute, None) is None:
        raise NotFittedError(
            f"{type(obj).__name__} is not fitted yet: call fit() or "
            f"partial_fit() with enough data before using {what}."
        )


def check_positive_int(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return int(value)
