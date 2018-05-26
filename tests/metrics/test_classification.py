import pytest

import dask.array as da
import sklearn.metrics

import dask_ml.metrics


@pytest.fixture(params=['accuracy_score'])
def metric_pairs(request):
    """Pairs of (dask-ml, sklearn) accuracy metrics.

    * accuracy_score
    """
    return (
        getattr(dask_ml.metrics, request.param),
        getattr(sklearn.metrics, request.param)
    )


@pytest.fixture(params=[True, False])
def normalize(request):
    """Boolean flag for normalize"""
    return request.param


@pytest.mark.parametrize('size', [(100,), (100, 2)])
def test_ok(size, metric_pairs, normalize):
    m1, m2 = metric_pairs

    if len(size) == 1:
        hi = 3
    else:
        hi = 1
    a = da.random.random_integers(0, hi, size=size, chunks=25)
    b = da.random.random_integers(0, hi, size=size, chunks=25)

    result = m1(a, b, normalize=normalize)
    expected = m2(a, b, normalize=normalize)
    assert abs(result - expected) < 1e-5
