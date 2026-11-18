

from __future__ import annotations

import csv
from typing import Iterator, Sequence

import numpy as np

__all__ = ["Frame", "read_csv", "iter_csv", "write_csv", "MISSING_TOKENS"]

MISSING_TOKENS = frozenset({"", "nan", "na", "n/a", "null", "none", "?"})


class Frame:
   

    def __init__(self, columns: dict):
        lengths = {len(v) for v in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"All columns must have the same length, got {sorted(lengths)}.")
        self._cols = {k: np.asarray(v) for k, v in columns.items()}

    @property
    def columns(self) -> list:
        return list(self._cols)

    @property
    def shape(self) -> tuple:
        return (len(self), len(self._cols))

    def __len__(self) -> int:
        return len(next(iter(self._cols.values()))) if self._cols else 0

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._cols:
                raise KeyError(f"Unknown column {key!r}; available: {self.columns}.")
            return self._cols[key]
        if isinstance(key, (list, tuple)):
            return Frame({k: self[k] for k in key})
        return Frame({k: v[key] for k, v in self._cols.items()})

    def __repr__(self) -> str:
        return f"Frame(n_rows={len(self)}, columns={self.columns})"

    def numeric_columns(self) -> list:
        return [k for k, v in self._cols.items() if v.dtype.kind == "f"]

    def categorical_columns(self) -> list:
        return [k for k, v in self._cols.items() if v.dtype.kind in "US"]

    def to_array(self, cols: Sequence[str] | None = None, dtype=None) -> np.ndarray:
        
        cols = self.columns if cols is None else list(cols)
        arrays = [self[c] for c in cols]
        if dtype is None:
            dtype = float if all(a.dtype.kind == "f" for a in arrays) else str
        return np.column_stack([a.astype(dtype) for a in arrays]) if arrays \
            else np.empty((len(self), 0))

    def iter_chunks(self, chunk_size: int) -> Iterator["Frame"]:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        for start in range(0, len(self), chunk_size):
            yield self[start:start + chunk_size]


def _convert_column(values: list) -> np.ndarray:
    cleaned = [v.strip() for v in values]
    missing = np.array([v.lower() in MISSING_TOKENS for v in cleaned], dtype=bool)
    try:
        out = np.array([np.nan if m else float(v) for v, m in zip(cleaned, missing)],
                       dtype=float)
        if np.isinf(out).any():
            raise ValueError("inf")
        return out
    except ValueError:
        return np.array(["" if m else v for v, m in zip(cleaned, missing)], dtype=str)


def _rows_to_frame(header: list, rows: list, usecols, path) -> Frame:
    idx = list(range(len(header))) if usecols is None else [header.index(c) for c in usecols]
    for line_no, row in rows:
        if len(row) != len(header):
            raise ValueError(
                f"{path}: line {line_no} has {len(row)} fields, expected {len(header)}."
            )
    cols = {}
    for i in idx:
        cols[header[i]] = _convert_column([r[i] for _, r in rows])
    return Frame(cols)


def _open_reader(path, delimiter):
    fh = open(path, "r", newline="", encoding="utf-8")
    reader = csv.reader(fh, delimiter=delimiter)
    try:
        header = [h.strip() for h in next(reader)]
    except StopIteration:
        fh.close()
        raise ValueError(f"{path} is empty: no header row found.") from None
    if len(set(header)) != len(header):
        fh.close()
        raise ValueError(f"{path} has duplicate column names: {header}.")
    return fh, reader, header


def _check_usecols(usecols, header, path):
    if usecols is None:
        return None
    unknown = [c for c in usecols if c not in header]
    if unknown:
        raise KeyError(f"{path}: columns {unknown} not found; header is {header}.")
    return list(usecols)


def read_csv(path, *, usecols: Sequence[str] | None = None, delimiter: str = ",") -> Frame:
   
    fh, reader, header = _open_reader(path, delimiter)
    with fh:
        usecols = _check_usecols(usecols, header, path)
        rows = [(i + 2, r) for i, r in enumerate(reader) if r]
    return _rows_to_frame(header, rows, usecols, path)


def iter_csv(path, chunk_size: int, *, usecols: Sequence[str] | None = None,
             delimiter: str = ",") -> Iterator[Frame]:
    
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    fh, reader, header = _open_reader(path, delimiter)
    with fh:
        usecols = _check_usecols(usecols, header, path)
        rows = []
        for i, r in enumerate(reader):
            if not r:
                continue
            rows.append((i + 2, r))
            if len(rows) == chunk_size:
                yield _rows_to_frame(header, rows, usecols, path)
                rows = []
        if rows:
            yield _rows_to_frame(header, rows, usecols, path)


def write_csv(path, data: dict, *, delimiter: str = ",", float_fmt: str = "{:.6g}") -> None:
   
    names = list(data)
    cols = [np.asarray(data[n]) for n in names]
    if len({len(c) for c in cols}) > 1:
        raise ValueError("All columns passed to write_csv must have the same length.")

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else float_fmt.format(v)
        return str(v)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow(names)
        for row in zip(*cols):
            w.writerow([fmt(v) for v in row])
