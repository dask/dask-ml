import numbers

import dask.array as da
import numpy as np
import pytest
import sklearn.metrics
from dask.array.utils import assert_eq

import dask_ml.metrics

_METRICS_TO_TEST = [
    "mean_squared_error",
    "mean_squared_log_error",
    "mean_absolute_error",
    "r2_score",
]

# mean_absolute_percentage_error() was added in scikit-learn 0.24.0
_METRICS_TO_TEST.append("mean_absolute_percentage_error")


@pytest.fixture(params=_METRICS_TO_TEST)
def metric_pairs(request):
    """Pairs of (dask-ml, sklearn) regression metrics.

    * mean_squared_error
    * mean_absolute_error
    * mean_absolute_percentage_error (if scikit-learn >= 0.24.0)
    * r2_score
    """
    return (
        getattr(dask_ml.metrics, request.param),
        getattr(sklearn.metrics, request.param),
    )


@pytest.mark.parametrize("compute", [True, False])
def test_ok(metric_pairs, compute):
    m1, m2 = metric_pairs

    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    result = m1(a, b, compute=compute)
    if compute:
        assert isinstance(result, numbers.Real)
    else:
        assert isinstance(result, da.Array)
    expected = m2(a, b)
    assert abs(result - expected) < 1e-5


@pytest.mark.skip(reason="FutureWarning: 'squared' is deprecated")
@pytest.mark.parametrize("squared", [True, False])
def test_mse_squared(squared):
    m1 = dask_ml.metrics.mean_squared_error
    m2 = sklearn.metrics.mean_squared_error

    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    result = m1(a, b, squared=squared)
    expected = m2(a, b, squared=squared)
    assert abs(result - expected) < 1e-5


@pytest.mark.skip(
    reason="InvalidParameterError: The 'multioutput' parameter of mean_squared_error "
    + "must be a string among..."
)
@pytest.mark.parametrize("multioutput", ["uniform_average", None])
def test_regression_metrics_unweighted_average_multioutput(metric_pairs, multioutput):
    m1, m2 = metric_pairs

    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    result = m1(a, b, multioutput=multioutput)
    expected = m2(a, b, multioutput=multioutput)
    assert abs(result - expected) < 1e-5


@pytest.mark.parametrize("compute", [True, False])
def test_regression_metrics_raw_values(metric_pairs, compute):
    m1, m2 = metric_pairs

    if m1.__name__ == "r2_score":
        pytest.skip("r2_score does not support multioutput='raw_values'")

    a = da.random.uniform(size=(100, 3), chunks=(25, 3))
    b = da.random.uniform(size=(100, 3), chunks=(25, 3))

    result = m1(a, b, multioutput="raw_values", compute=compute)
    expected = m2(a, b, multioutput="raw_values")

    if compute:
        assert isinstance(result, np.ndarray)
    else:
        assert isinstance(result, da.Array)

    assert_eq(result, expected)
    assert result.shape == (3,)


def test_regression_metrics_do_not_support_weighted_multioutput(metric_pairs):
    m1, _ = metric_pairs

    a = da.random.uniform(size=(100, 3), chunks=(25, 3))
    b = da.random.uniform(size=(100, 3), chunks=(25, 3))
    weights = da.random.uniform(size=(3,))

    if m1.__name__ == "r2_score":
        error_msg = "'multioutput' must be 'uniform_average'"
    else:
        error_msg = "Weighted 'multioutput' not supported."

    with pytest.raises((NotImplementedError, ValueError), match=error_msg):
        _ = m1(a, b, multioutput=weights)
