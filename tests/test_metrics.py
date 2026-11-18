
import numpy as np
import pytest

from numcompute_stream.metrics import (ROCAUC, Accuracy, ConfusionMatrix, F1Score, Precision,
                                       Recall, RollingMetric, accuracy, f1_score,
                                       roc_auc_score)
from benchmark.loop_impl import confusion_matrix_loop


def test_accumulating_equals_one_shot(rng):
    yt = rng.integers(0, 3, 1000)
    yp = np.where(rng.random(1000) < 0.7, yt, rng.integers(0, 3, 1000))
    metrics = [Accuracy(), Precision(), Recall(), F1Score(average="micro"),
               F1Score(average="weighted")]
    for s in range(0, 1000, 130):
        for m in metrics:
            m.update(yt[s:s + 130], yp[s:s + 130])
    assert metrics[0].result() == pytest.approx(np.mean(yt == yp))
    assert metrics[3].result() == pytest.approx(np.mean(yt == yp))   # micro-F1 == accuracy
    assert metrics[2].result() == pytest.approx(
        np.mean([np.mean(yp[yt == c] == c) for c in range(3)]))


def test_binary_precision_recall_f1_by_hand():
    yt = [1, 1, 1, 0, 0, 0, 0, 1]
    yp = [1, 1, 0, 1, 0, 0, 0, 0]       # TP=2 FP=1 FN=2
    assert Precision("binary").update(yt, yp).result() == pytest.approx(2 / 3)
    assert Recall("binary").update(yt, yp).result() == pytest.approx(0.5)
    assert f1_score(yt, yp, average="binary") == pytest.approx(4 / 7)


def test_confusion_matrix_grows_with_new_labels():
    cm = ConfusionMatrix()
    cm.update([0, 1], [0, 1])
    cm.update([2, 1], [1, 2])                       # label 2 appears late
    np.testing.assert_array_equal(cm.labels_, [0, 1, 2])
    expected = confusion_matrix_loop([0, 1, 2, 1], [0, 1, 1, 2], [0, 1, 2])
    np.testing.assert_array_equal(cm.result(), expected)


def test_confusion_matrix_string_labels_and_fixed_set():
    cm = ConfusionMatrix().update(["cat", "dog"], ["dog", "dog"])
    assert list(cm.labels_) == ["cat", "dog"] and cm.result()[0, 1] == 1
    with pytest.raises(ValueError, match="not in the fixed label set"):
        ConfusionMatrix(labels=[0, 1]).update([0, 2], [0, 1])


def test_reset_and_empty_results():
    m = Accuracy()
    assert np.isnan(m.result())                     # nothing seen yet
    m.update([1, 0], [1, 1])
    assert m.result() == 0.5
    assert np.isnan(m.reset().result())


def test_zero_division_is_controlled():
    p = Precision("binary", zero_division=0.0).update([1, 0], [0, 0]).result()
    assert p == 0.0
    p1 = Precision("binary", zero_division=1.0).update([1, 0], [0, 0]).result()
    assert p1 == 1.0


def test_rolling_window_sees_only_recent_samples():
    r = RollingMetric(Accuracy, window=4)
    r.update([1, 1, 1, 1], [1, 1, 1, 1])
    r.update([0, 0], [1, 1])                         # window = [1,1,0,0] vs [1,1,1,1]
    assert r.result() == 0.5
    cumulative = accuracy([1] * 4 + [0] * 2, [1] * 6)
    assert cumulative == pytest.approx(4 / 6)


def test_auc_binned_close_to_exact(rng):
    y = rng.integers(0, 2, 5000)
    s = np.clip(0.35 * y + rng.normal(0.3, 0.2, 5000), 0, 1)
    binned, exact = ROCAUC(n_bins=1000), ROCAUC(exact=True)
    for i in range(0, 5000, 700):
        binned.update(y[i:i + 700], s[i:i + 700])
        exact.update(y[i:i + 700], s[i:i + 700])
    assert binned.result() == pytest.approx(exact.result(), abs=2e-3)
    assert exact.result() == pytest.approx(roc_auc_score(y, s))


def test_auc_edge_cases():
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc_score([0, 1], [0.5, 0.5]) == 0.5             # tie = coin flip
    assert np.isnan(ROCAUC().update([1, 1], [0.2, 0.9]).result())  # one class: undefined
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ROCAUC().update([0, 1], [-1.0, 3.0])


def test_metric_input_validation():
    with pytest.raises(ValueError, match="different lengths"):
        Accuracy().update([1, 0, 1], [1, 0])
    with pytest.raises(ValueError, match="average"):
        F1Score(average="nope")
