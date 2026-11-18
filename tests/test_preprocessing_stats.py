

import numpy as np
import pytest

from numcompute_stream import (MinMaxScaler, NotFittedError, OneHotEncoder, SimpleImputer,
                               StandardScaler, StreamingHistogram, StreamingQuantile,
                               StreamingStats, update_stats)
from numcompute_stream.stats import chunked_mean, chunked_variance
from benchmark.loop_impl import welford_loop


def chunks(X, size):
    return [X[i:i + size] for i in range(0, len(X), size)]


@pytest.mark.parametrize("size", [1, 7, 100, 1000])
def test_streaming_mean_var_match_batch(rng, size):
    X = rng.normal(3, 2, size=(503, 4))
    s = StreamingStats()
    for c in chunks(X, size):
        s.update_stats(c)
    np.testing.assert_allclose(s.mean, X.mean(axis=0))
    np.testing.assert_allclose(s.var(), X.var(axis=0))
    np.testing.assert_allclose(s.var(ddof=1), X.var(axis=0, ddof=1))
    np.testing.assert_array_equal(s.min_, X.min(axis=0))


def test_streaming_stats_ignore_nan_per_column(rng):
    X = rng.normal(size=(200, 3))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[:, 2] = np.nan                                   
    s = update_stats(None, X[:50])
    s = update_stats(s, X[50:])
    np.testing.assert_allclose(s.mean[:2], np.nanmean(X[:, :2], axis=0))
    np.testing.assert_allclose(s.var()[:2], np.nanvar(X[:, :2], axis=0))
    assert np.isnan(s.mean[2]) and s.count_[2] == 0
    m, v = welford_loop(X)                             
    np.testing.assert_allclose(s.var()[:2], v[:2])


def test_variance_stable_with_large_offset(rng):
    X = 1e9 + rng.normal(size=(10_000, 1))
    s = StreamingStats()
    for c in chunks(X, 333):
        s.update_stats(c)
    assert s.var()[0] == pytest.approx(np.var(X - 1e9), rel=1e-6)
    naive = np.mean(X ** 2) - np.mean(X) ** 2        
    assert abs(naive - np.var(X - 1e9)) > 1.0


def test_merge_equals_single_pass(rng):
    X = rng.normal(size=(300, 2))
    a = StreamingStats().update_stats(X[:120])
    b = StreamingStats().update_stats(X[120:])
    np.testing.assert_allclose(a.merge(b).var(), X.var(axis=0))


def test_stats_reject_changed_width():
    s = StreamingStats().update_stats(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="features"):
        s.update_stats(np.zeros((3, 3)))


def test_chunked_helpers(rng):
    X = rng.normal(size=(90, 2))
    np.testing.assert_allclose(chunked_mean(chunks(X, 20)), X.mean(0))
    np.testing.assert_allclose(chunked_variance(chunks(X, 20), ddof=1), X.var(0, ddof=1))


def test_quantile_exact_below_capacity(rng):
    X = rng.normal(size=(400, 2))
    q = StreamingQuantile(capacity=1000)
    for c in chunks(X, 37):
        q.update_stats(c)
    assert q.is_exact
    np.testing.assert_allclose(q.quantile([0.1, 0.5, 0.9]), np.quantile(X, [0.1, 0.5, 0.9], axis=0))


def test_quantile_reservoir_estimate(rng):
    x = rng.uniform(0, 1, size=(50_000, 1))
    q = StreamingQuantile(capacity=4000, random_state=0)
    for c in chunks(x, 1000):
        q.update_stats(c)
    assert not q.is_exact
    assert q.median()[0] == pytest.approx(0.5, abs=0.03)
    with pytest.raises(ValueError):
        q.quantile(1.5)


def test_histogram_counts_and_sliding_window():
    h = StreamingHistogram(bins=4, range=(0, 4), window=2)
    h.update_stats([0.5, 1.5])
    h.update_stats([2.5, 2.6])
    h.update_stats([3.5, 9.0, np.nan])               
    np.testing.assert_array_equal(h.counts, [0, 0, 2, 1])   
    assert h.overflow_ == 1 and h.n_missing_ == 1


def test_histogram_zero_variance_first_chunk():
    h = StreamingHistogram(bins=2).update_stats([5.0, 5.0, 5.0])
    assert h.edges_[0] < 5 < h.edges_[-1] and h.counts.sum() == 3


def test_standard_scaler_partial_equals_fit(rng):
    X = rng.normal(10, 5, size=(500, 3))
    inc = StandardScaler()
    for c in chunks(X, 64):
        inc.partial_fit(c)
    full = StandardScaler().fit(X)
    np.testing.assert_allclose(inc.transform(X), full.transform(X))
    np.testing.assert_allclose(full.transform(X).mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(full.inverse_transform(full.transform(X)), X)


def test_standard_scaler_zero_variance_and_nan():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [np.nan, 5.0]])
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_array_equal(Z[:, 1], 0.0)      
    assert np.isnan(Z[2, 0]) and np.isfinite(Z[:2]).all()


def test_scaler_errors():
    with pytest.raises(NotFittedError):
        StandardScaler().transform([[1.0]])
    s = StandardScaler().fit(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="2 features"):
        s.transform(np.zeros((3, 3)))


def test_minmax_scaler_streaming(rng):
    X = rng.uniform(-3, 7, size=(200, 2))
    m = MinMaxScaler()
    for c in chunks(X, 30):
        m.partial_fit(c)
    Z = m.transform(X)
    np.testing.assert_allclose(Z.min(0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.max(0), 1)
    assert np.all(MinMaxScaler().fit([[2.0], [2.0]]).transform([[2.0]]) == 0)


def test_imputer_mean_updates_on_the_fly():
    imp = SimpleImputer("mean")
    imp.partial_fit([[1.0], [np.nan]])
    assert imp.transform([[np.nan]])[0, 0] == 1.0
    imp.partial_fit([[3.0], [5.0]])
    assert imp.transform([[np.nan]])[0, 0] == 3.0   


@pytest.mark.parametrize("strategy,expected", [("median", 2.0), ("most_frequent", 1.0),
                                               ("constant", -1.0)])
def test_imputer_strategies(strategy, expected):
    X = np.array([[1.0], [1.0], [2.0], [np.nan], [7.0], [9.0]])
    imp = SimpleImputer(strategy, fill_value=-1.0)
    for c in chunks(X, 2):
        imp.partial_fit(c)
    assert imp.transform([[np.nan]])[0, 0] == expected


def test_imputer_all_missing_column_falls_back():
    imp = SimpleImputer(fill_value=0.0).fit([[np.nan, 1.0], [np.nan, 3.0]])
    np.testing.assert_array_equal(imp.empty_columns_, [0])
    np.testing.assert_array_equal(imp.transform([[np.nan, np.nan]]), [[0.0, 2.0]])


def test_most_frequent_tie_is_order_independent():
    a = SimpleImputer("most_frequent").fit([[5.0], [3.0]])
    b = SimpleImputer("most_frequent").fit([[3.0], [5.0]])
    assert a.statistics_[0] == b.statistics_[0] == 3.0


def test_onehot_expands_incrementally_append_only():
    enc = OneHotEncoder()
    enc.partial_fit([["red"], ["blue"]])
    first = enc.transform([["red"], ["blue"]])
    enc.partial_fit([["green"], ["red"]])
    later = enc.transform([["red"], ["blue"], ["green"]])
    assert enc.categories_ == [["red", "blue", "green"]]
    np.testing.assert_array_equal(later[:2, :2], first)       
    np.testing.assert_array_equal(later[2], [0, 0, 1])
    assert enc.get_feature_names_out(["colour"]) == ["colour=red", "colour=blue", "colour=green"]


def test_onehot_unknown_missing_and_cap():
    enc = OneHotEncoder(max_categories=2).fit([["a", "x"], ["b", "x"], ["c", ""]])
    assert enc.categories_ == [["a", "b"], ["x"]]
    np.testing.assert_array_equal(enc.transform([["zzz", ""]]), [[0, 0, 0]])
    strict = OneHotEncoder(handle_unknown="error").fit([["a"]])
    with pytest.raises(ValueError, match="Unknown categories"):
        strict.transform([["b"]])
