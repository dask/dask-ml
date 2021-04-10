import numbers

import dask.array as da
import pytest
import sklearn.metrics

import dask_ml.metrics
from dask_ml._compat import SK_024

_METRICS_TO_TEST = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
]

# mean_absolute_percentage_error() was added in scikit-learn 0.24.0
if SK_024:
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


@pytest.mark.parametrize("squared", [True, False])
def test_mse_squared(squared):
    m1 = dask_ml.metrics.mean_squared_error
    m2 = sklearn.metrics.mean_squared_error

    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    result = m1(a, b, squared=squared)
    expected = m2(a, b, squared=squared)
    assert abs(result - expected) < 1e-5


def test_mean_squared_log_error():
    m1 = dask_ml.metrics.mean_squared_log_error
    m2 = sklearn.metrics.mean_squared_log_error

    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    result = m1(a, b)
    expected = m2(a, b)
    assert abs(result - expected) < 1e-5
