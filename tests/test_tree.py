
import numpy as np
import pytest

from numcompute_stream import DecisionTreeClassifier, NotFittedError
from numcompute_stream.tree import best_split, entropy, gini
from benchmark.loop_impl import best_split_loop


def test_gini_and_entropy_known_values():
    assert gini([[6, 6]])[0] == pytest.approx(0.5)
    assert entropy([[6, 6]])[0] == pytest.approx(1.0)
    assert gini([[0, 7]])[0] == 0.0 and entropy([[0, 7]])[0] == 0.0
    assert gini([[0, 0]])[0] == 0.0          


def test_root_gain_matches_lecture_example(students):
    X, y = students
    s = best_split(X, y, 2, "gini")
    assert s.feature == 0                     
    assert s.gain == pytest.approx(0.248, abs=1e-3)
    assert (s.n_left, s.n_right) == (5, 7)


@pytest.mark.parametrize("depth,acc", [(1, 10 / 12), (2, 11 / 12), (3, 1.0)])
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_depth_accuracy_matches_lecture(students, depth, acc, criterion):
    X, y = students
    t = DecisionTreeClassifier(max_depth=depth, criterion=criterion).fit(X, y)
    assert t.score(X, y) == pytest.approx(acc)


def test_vectorised_split_equals_loop_reference(rng):
    for _ in range(5):
        X = np.round(rng.normal(size=(120, 4)), 1)      
        y = rng.integers(0, 3, 120)
        a = best_split(X, y, 3, "entropy")
        b = best_split_loop(X, y, 3, "entropy")
        assert (a.feature, a.threshold) == (b[0], b[1])
        assert a.gain == pytest.approx(b[2])


def test_learns_axis_aligned_rule(blobs):
    X, y = blobs
    t = DecisionTreeClassifier(max_depth=4).fit(X[:900], y[:900])
    assert t.score(X[900:], y[900:]) > 0.97
    assert set(np.flatnonzero(t.feature_importances_ > 0.05)) == {0, 1}


def test_depth_limit_respected(blobs):
    X, y = blobs
    for d in (0, 1, 3):
        assert DecisionTreeClassifier(max_depth=d).fit(X, y).get_depth() <= d


def test_max_depth_zero_predicts_majority():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.array([0] * 7 + [1] * 3)
    t = DecisionTreeClassifier(max_depth=0).fit(X, y)
    assert t.get_n_leaves() == 1 and np.all(t.predict(X) == 0)


def test_min_samples_split_stops_growth(blobs):
    X, y = blobs
    small = DecisionTreeClassifier(max_depth=8, min_samples_split=600).fit(X, y)
    big = DecisionTreeClassifier(max_depth=8).fit(X, y)
    assert small.get_n_leaves() < big.get_n_leaves()


def test_predict_proba_rows_sum_to_one(blobs):
    X, y = blobs
    p = DecisionTreeClassifier(max_depth=3).fit(X, y).predict_proba(X)
    assert p.shape == (len(X), 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_string_labels_and_multiclass(rng):
    X = rng.normal(size=(300, 2))
    y = np.where(X[:, 0] > 0.5, "high", np.where(X[:, 0] < -0.5, "low", "mid"))
    t = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert list(t.classes_) == ["high", "low", "mid"]
    assert t.score(X, y) == 1.0


def test_max_features_is_reproducible(blobs):
    X, y = blobs
    a = DecisionTreeClassifier(max_depth=5, max_features="sqrt", random_state=1).fit(X, y)
    b = DecisionTreeClassifier(max_depth=5, max_features="sqrt", random_state=1).fit(X, y)
    np.testing.assert_array_equal(a.feature_, b.feature_)


def test_single_class_gives_single_leaf():
    X = np.random.default_rng(0).normal(size=(50, 3))
    t = DecisionTreeClassifier().fit(X, np.ones(50, dtype=int))
    assert t.get_n_leaves() == 1 and np.all(t.predict(X) == 1)


def test_zero_variance_features_make_a_leaf():
    X = np.ones((40, 3))
    y = np.array([0, 1] * 20)
    t = DecisionTreeClassifier().fit(X, y)
    assert t.get_n_leaves() == 1
    assert t.predict(X[:1])[0] == 0          


def test_tie_breaking_is_deterministic_and_order_free():
    X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)  
    y = np.array([0, 0, 1, 1])
    t = DecisionTreeClassifier().fit(X, y)
    assert t.feature_[0] == 0                 
    perm = [3, 1, 2, 0]
    t2 = DecisionTreeClassifier().fit(X[perm], y[perm])
    assert t2.feature_[0] == 0 and t2.threshold_[0] == t.threshold_[0]


def test_nan_features_learn_a_direction():
    x = np.array([1, 2, 3, np.nan, np.nan, 10, 11, 12], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    t = DecisionTreeClassifier(max_depth=1).fit(x.reshape(-1, 1), y)
    assert t.score(x.reshape(-1, 1), y) == 1.0
    assert not t.missing_left_[0]            
    assert t.predict([[np.nan]])[0] == 1


def test_threshold_between_adjacent_floats():
    a = 1.0
    b = np.nextafter(a, 2.0)
    X = np.array([[a], [a], [b], [b]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTreeClassifier().fit(X, y)
    assert t.score(X, y) == 1.0


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        DecisionTreeClassifier().predict([[1.0]])


@pytest.mark.parametrize("kwargs,err", [
    ({"max_depth": -1}, ValueError), ({"criterion": "mse"}, ValueError),
    ({"buffer_size": 0}, ValueError), ({"max_features": 1.5}, ValueError),
    ({"max_depth": 2.5}, TypeError), ({"min_samples_split": 1}, ValueError),
])
def test_bad_hyperparameters_raise(kwargs, err):
    with pytest.raises(err):
        DecisionTreeClassifier(**kwargs)


def test_shape_errors_are_informative(blobs):
    X, y = blobs
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        DecisionTreeClassifier().fit(X, y[:-1])
    t = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        t.predict(X[:, :2])
    with pytest.raises(TypeError, match="numeric"):
        t.predict([["a", "b", "c", "d"]])
    with pytest.raises(ValueError, match="NaN labels"):
        DecisionTreeClassifier().fit(X[:3], [0, np.nan, 1])


def test_partial_fit_waits_for_buffer(blobs):
    X, y = blobs
    t = DecisionTreeClassifier(buffer_size=500)
    t.partial_fit(X[:200], y[:200])
    t.partial_fit(X[200:400], y[200:400])
    assert t.n_retrains_ == 0 and t.buffer_fill == 400
    with pytest.raises(NotFittedError):
        t.predict(X[:5])
    t.partial_fit(X[400:600], y[400:600])     
    assert t.n_retrains_ == 1 and t.buffer_fill == 0
    assert t.n_node_samples_[0] == 600        


def test_prediction_frozen_between_retrains(blobs):
    X, y = blobs
    t = DecisionTreeClassifier(buffer_size=300).partial_fit(X[:300], y[:300])
    before = t.predict(X)
    t.partial_fit(X[300:500], 1 - y[300:500])
    np.testing.assert_array_equal(t.predict(X), before)


def test_partial_fit_matches_fit_on_same_buffer(blobs):
    X, y = blobs
    stream = DecisionTreeClassifier(max_depth=4, buffer_size=400)
    for s in range(0, 400, 100):
        stream.partial_fit(X[s:s + 100], y[s:s + 100])
    batch = DecisionTreeClassifier(max_depth=4).fit(X[:400], y[:400])
    np.testing.assert_array_equal(stream.predict(X), batch.predict(X))


def test_chunk_larger_than_buffer_and_flush(blobs):
    X, y = blobs
    t = DecisionTreeClassifier(buffer_size=100).partial_fit(X[:350], y[:350])
    assert t.n_retrains_ == 1 and t.n_node_samples_[0] == 350
    t.partial_fit(X[350:380], y[350:380]).flush()
    assert t.n_retrains_ == 2 and t.buffer_fill == 0
    t.flush()                                 
    assert t.n_retrains_ == 2


def test_new_class_appears_mid_stream():
    X = np.arange(30.0).reshape(-1, 1)
    t = DecisionTreeClassifier(buffer_size=10)
    t.partial_fit(X[:10], np.r_[np.zeros(5), np.ones(5)].astype(int))
    t.partial_fit(X[10:12], [2, 2])           
    assert list(t.classes_) == [0, 1, 2]
    p = t.predict_proba(X[:3])
    assert p.shape == (3, 3) and np.all(p[:, 2] == 0)


def test_growing_feature_width_is_padded():
    t = DecisionTreeClassifier(buffer_size=6)
    t.partial_fit(np.array([[0, 1], [1, 0], [0, 1]], float), [0, 1, 0])
    t.partial_fit(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], float), [1, 0, 1])
    assert t.n_features_in_ == 3 and t.n_retrains_ == 1
    with pytest.raises(ValueError, match="never shrink"):
        t.partial_fit(np.zeros((2, 2)), [0, 1])


def test_export_text_and_memory(students):
    X, y = students
    t = DecisionTreeClassifier(max_depth=2).fit(X, y)
    text = t.export_text(["submitted", "attended", "study_group"])
    assert "submitted <= 0.5" in text and "class: 1" in text
    assert t.memory_footprint() > 0
