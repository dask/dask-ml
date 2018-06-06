import pytest

import dask.array as da
import sklearn.metrics

import dask_ml.metrics


@pytest.fixture(params=[
    'mean_squared_error', 'mean_absolute_error', 'r2_score',
])
def metric_pairs(request):
    """Pairs of (dask-ml, sklearn) regression metrics.

    * mean_squared_error
    * mean_absolute_error
    * r2_score
    """
    return (
        getattr(dask_ml.metrics, request.param),
        getattr(sklearn.metrics, request.param)
    )


def test_ok(metric_pairs):
    m1, m2 = metric_pairs

    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    result = m1(a, b)
    expected = m2(a, b)
    assert abs(result - expected) < 1e-5
