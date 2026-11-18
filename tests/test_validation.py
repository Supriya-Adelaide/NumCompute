
import numpy as np
import pytest

from numcompute_stream._validation import check_array, check_positive_int, check_y
from numcompute_stream import DecisionTreeClassifier, StandardScaler


def test_1d_input_becomes_single_column():
    assert check_array([1, 2, 3]).shape == (3, 1)


@pytest.mark.parametrize("bad,err,msg", [
    (np.zeros((2, 2, 2)), ValueError, "2-D"),
    (5.0, ValueError, "scalar"),
    ([[1.0, np.inf]], ValueError, "infinite"),
    (np.empty((0, 3)), ValueError, "at least 1"),
    (None, TypeError, "None"),
])
def test_check_array_rejects(bad, err, msg):
    with pytest.raises(err, match=msg):
        check_array(bad)


def test_nan_policy_and_labels():
    with pytest.raises(ValueError, match="impute"):
        check_array([[np.nan]], allow_nan=False)
    assert check_y(np.array([[1.0], [0.0]])).dtype.kind == "i"     
    with pytest.raises(ValueError, match="1-D"):
        check_y(np.zeros((2, 2)))


def test_positive_int_rejects_bool_and_float():
    with pytest.raises(TypeError):
        check_positive_int(True, "n")
    with pytest.raises(TypeError):
        check_positive_int(2.0, "n")


def test_inf_rejected_end_to_end():
    with pytest.raises(ValueError, match="infinite"):
        StandardScaler().partial_fit([[1.0], [np.inf]])
    with pytest.raises(ValueError, match="infinite"):
        DecisionTreeClassifier().fit([[1.0], [-np.inf]], [0, 1])
