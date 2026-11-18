
import os
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def students():
    
    rows = np.array([
        [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 0, 0], [1, 0, 0, 0],
        [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0],
    ])
    return rows[:, :3].astype(float), rows[:, 3]


@pytest.fixture
def blobs(rng):
    X = rng.normal(size=(1200, 4))
    y = ((X[:, 0] > 0.2) & (X[:, 1] < 0.5)).astype(int)
    return X, y
