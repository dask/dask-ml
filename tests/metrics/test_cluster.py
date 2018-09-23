import numbers

import dask.array as da
import numpy as np
import packaging.version
import pytest
import sklearn.metrics
import random

import dask_ml.metrics
from dask_ml._compat import DASK_VERSION


@pytest.fixture(params=["mutual_info_score"])
def metric_pairs(request):
    """Pairs of (dask-ml, sklearn) accuracy metrics.

    * accuracy_score
    """
    return (
        getattr(dask_ml.metrics, request.param),
        getattr(sklearn.metrics, request.param),
    )

@pytest.mark.parametrize("size", [(100,), (200,), (300,),(400,),(500,),(1000,)])
def test_ok(size, metric_pairs):
    m1, m2 = metric_pairs

    hi = random.randint(2,9)
    a = da.random.random_integers(0, hi, size=size, chunks=25)
    b = da.random.random_integers(0, hi, size=size, chunks=25)

    expected = m2(a, b)
    result   = m1(a, b)
    assert abs(result - expected) < 1e-5
