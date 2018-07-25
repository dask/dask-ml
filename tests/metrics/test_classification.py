import numbers

import dask.array as da
import numpy as np
import packaging.version
import pytest
import sklearn.metrics

import dask_ml.metrics
from dask_ml._compat import DASK_VERSION


@pytest.fixture(params=["accuracy_score"])
def metric_pairs(request):
    """Pairs of (dask-ml, sklearn) accuracy metrics.

    * accuracy_score
    """
    return (
        getattr(dask_ml.metrics, request.param),
        getattr(sklearn.metrics, request.param),
    )


@pytest.fixture(params=[True, False])
def normalize(request):
    """Boolean flag for normalize"""
    return request.param


@pytest.mark.parametrize("size", [(100,), (100, 2)])
@pytest.mark.parametrize("compute", [True, False])
def test_ok(size, metric_pairs, normalize, compute):
    m1, m2 = metric_pairs

    if len(size) == 1:
        hi = 3
    else:
        hi = 1
    a = da.random.random_integers(0, hi, size=size, chunks=25)
    b = da.random.random_integers(0, hi, size=size, chunks=25)

    result = m1(a, b, normalize=normalize, compute=compute)
    if compute:
        assert isinstance(result, numbers.Real)
    else:
        assert isinstance(result, da.Array)
    expected = m2(a, b, normalize=normalize)
    assert abs(result - expected) < 1e-5


@pytest.mark.skipif(
    DASK_VERSION <= packaging.version.parse("0.18.0"),
    reason="Requires dask.array.average",
)
def test_sample_weight(metric_pairs, normalize):
    m1, m2 = metric_pairs

    size = (100,)
    a = da.random.random_integers(0, 3, size=size, chunks=25)
    b = da.random.random_integers(0, 3, size=size, chunks=25)

    sample_weight_np = np.random.random_sample(size[0])
    sample_weight_da = da.from_array(sample_weight_np, chunks=25)

    result = m1(a, b, sample_weight=sample_weight_da, normalize=normalize, compute=True)
    expected = m2(a, b, sample_weight=sample_weight_np, normalize=normalize)
    assert abs(result - expected) < 1e-5


@pytest.mark.skipif(
    DASK_VERSION > packaging.version.parse("0.18.0"),
    reason="Requires dask.array.average to be missing",
)
def test_sample_weight_raises(metric_pairs, normalize):
    m1, m2 = metric_pairs

    size = (100,)
    a = da.random.random_integers(0, 3, size=size, chunks=25)
    b = da.random.random_integers(0, 3, size=size, chunks=25)

    sample_weight_np = np.random.random_sample(size[0])
    sample_weight_da = da.from_array(sample_weight_np, chunks=25)

    with pytest.raises(NotImplementedError):
        m1(a, b, sample_weight=sample_weight_da, normalize=normalize, compute=True)
