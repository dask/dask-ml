import pytest
import numpy as np
import dask.array as da

from dask_glm import utils
from dask.array.utils import assert_eq


def test_add_intercept():
    X = np.zeros((4, 4))
    result = utils.add_intercept(X)
    expected = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ], dtype=X.dtype)
    assert_eq(result, expected)


def test_add_intercept_dask():
    X = da.from_array(np.zeros((4, 4)), chunks=(2, 4))
    result = utils.add_intercept(X)
    expected = da.from_array(np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ], dtype=X.dtype), chunks=2)
    assert_eq(result, expected)


def test_sparse():
    sparse = pytest.importorskip('sparse')
    from sparse.utils import assert_eq
    x = sparse.COO({(0, 0): 1, (1, 2): 2, (2, 1): 3})
    y = x.todense()
    assert utils.sum(x) == utils.sum(x.todense())
    for func in [utils.sigmoid, utils.sum, utils.exp]:
        assert_eq(func(x), func(y))
