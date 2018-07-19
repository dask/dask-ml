import numbers

import packaging.version
import dask.array as da
import numpy as np
import pytest
import sklearn.metrics

import dask_ml.metrics
from dask_ml._compat import DASK_VERSION


@pytest.fixture(params=["mean_squared_error", "mean_absolute_error", "r2_score"])
def metric_pairs(request):
    """Pairs of (dask-ml, sklearn) regression metrics.

    * mean_squared_error
    * mean_absolute_error
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


@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
def test_multioutput(metric_pairs, multioutput):
    m1, m2 = metric_pairs

    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    result = m1(a, b, multioutput=multioutput, compute=True)
    expected = m2(a, b, multioutput=multioutput)
    assert abs(result - expected) < 1e-5


def test_variance_weighted_multioutput():
    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    result = dask_ml.metrics.r2_score(
        a, b, multioutput="variance_weighted", compute=True
    )
    expected = sklearn.metrics.r2_score(a, b, multioutput="variance_weighted")
    assert abs(result - expected) < 1e-5


@pytest.mark.skipif(
    DASK_VERSION <= packaging.version.parse("0.18.0"),
    reason="Requires dask.array.average",
)
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
def test_sample_weight(metric_pairs, multioutput):
    m1, m2 = metric_pairs

    size = (100,)
    a = da.random.uniform(size=(100,), chunks=(25,))
    b = da.random.uniform(size=(100,), chunks=(25,))

    sample_weight_np = np.random.random_sample(size[0])
    sample_weight_da = da.from_array(sample_weight_np, chunks=25)

    result = m1(
        a, b, multioutput=multioutput, sample_weight=sample_weight_da, compute=True
    )
    expected = m2(a, b, multioutput=multioutput, sample_weight=sample_weight_np)
    assert abs(result - expected) < 1e-5
