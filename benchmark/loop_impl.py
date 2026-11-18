

import math

import numpy as np


def _impurity(counts, criterion):
    n = sum(counts)
    if n == 0:
        return 0.0
    if criterion == "gini":
        return 1.0 - sum((c / n) ** 2 for c in counts)
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0)


def best_split_loop(X, y_idx, n_classes, criterion="gini"):
    
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    parent_counts = [0] * n_classes
    for label in y_idx:
        parent_counts[int(label)] += 1
    parent = _impurity(parent_counts, criterion)
    best = None
    best_gain = 1e-12
    for f in range(d):
        values = sorted(set(float(v) for v in X[:, f]))
        for a, b in zip(values[:-1], values[1:]):
            t = a + (b - a) / 2.0
            if not t < b:
                t = a
            left = [0] * n_classes
            right = [0] * n_classes
            for i in range(n):
                if X[i, f] <= t:
                    left[int(y_idx[i])] += 1
                else:
                    right[int(y_idx[i])] += 1
            nl, nr = sum(left), sum(right)
            g = parent - (nl * _impurity(left, criterion) + nr * _impurity(right, criterion)) / n
            if g > best_gain + 1e-12:
                best, best_gain = (f, t, g), g
    return best


def welford_loop(X):
    X = np.asarray(X, dtype=float)
    n_rows, d = X.shape
    count = [0] * d
    mean = [0.0] * d
    m2 = [0.0] * d
    for i in range(n_rows):
        for j in range(d):
            x = X[i, j]
            if x != x:          
                continue
            count[j] += 1
            delta = x - mean[j]
            mean[j] += delta / count[j]
            m2[j] += delta * (x - mean[j])
    var = [m2[j] / count[j] if count[j] else float("nan") for j in range(d)]
    return np.array(mean), np.array(var)


def confusion_matrix_loop(y_true, y_pred, labels):
    pos = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    m = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        m[pos[t]][pos[p]] += 1
    return np.array(m)


def predict_loop(tree, X):
    X = np.asarray(X, dtype=float)
    out = []
    for row in X:
        node = 0
        while tree.feature_[node] >= 0:
            v = row[tree.feature_[node]]
            if v != v:
                go_left = tree.missing_left_[node]
            else:
                go_left = v <= tree.threshold_[node]
            node = tree.left_[node] if go_left else tree.right_[node]
        out.append(tree._fit_classes[int(np.argmax(tree.value_[node]))])
    return np.array(out)
