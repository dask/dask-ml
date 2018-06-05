import packaging.version
import pytest
import dask
import dask.array as da
import numpy as np
import numpy.testing as npt
import sklearn
import sklearn.metrics as sm
from dask.array.utils import assert_eq

import dask_ml.metrics as dm
from dask_ml._compat import SK_VERSION, dummy_context


def test_pairwise_distances(X_blobs):
    centers = X_blobs[::100].compute()
    result = dm.pairwise_distances(X_blobs, centers)
    expected = sm.pairwise_distances(X_blobs.compute(), centers)
    assert_eq(result, expected, atol=1e-4)


def test_pairwise_distances_argmin_min(X_blobs):
    centers = X_blobs[::100].compute()

    if SK_VERSION >= packaging.version.parse("0.20.0.dev0"):
        # X_blobs has 500 rows per block.
        # Ensure 500 rows in the scikit-learn version too.
        working_memory = 80 * 500 / 2**20

        ctx = sklearn.config_context(working_memory=working_memory)
    else:
        ctx = dummy_context()

    with ctx:
        a_, b_ = sm.pairwise_distances_argmin_min(X_blobs.compute(), centers)
        a, b = dm.pairwise_distances_argmin_min(X_blobs, centers)
        a, b = dask.compute(a, b)

    npt.assert_array_equal(a, a_)
    npt.assert_array_equal(b, b_)


def test_euclidean_distances():
    X = da.random.uniform(size=(100, 4), chunks=50)
    Y = da.random.uniform(size=(100, 4), chunks=50)
    a = dm.euclidean_distances(X, Y)
    b = sm.euclidean_distances(X, Y)
    assert_eq(a, b)

    x_norm_squared = (X ** 2).sum(axis=1).compute()[:, np.newaxis]
    a = dm.euclidean_distances(X, Y, X_norm_squared=x_norm_squared)
    b = sm.euclidean_distances(X, Y, X_norm_squared=x_norm_squared)
    assert_eq(a, b)

    y_norm_squared = (Y ** 2).sum(axis=1).compute()[np.newaxis, :]
    a = dm.euclidean_distances(X, Y, Y_norm_squared=y_norm_squared)
    b = sm.euclidean_distances(X, Y, Y_norm_squared=y_norm_squared)
    assert_eq(a, b)


def test_euclidean_distances_same():
    X = da.random.uniform(size=(100, 4), chunks=50)
    a = dm.euclidean_distances(X, X)
    b = sm.euclidean_distances(X, X)
    assert_eq(a, b, atol=1e-4)

    x_norm_squared = (X ** 2).sum(axis=1).compute()[:, np.newaxis]
    assert_eq(X, X, Y_norm_squared=x_norm_squared, atol=1e-4)


@pytest.mark.parametrize('kernel', [
    'linear',
    'polynomial',
    'rbf',
    'sigmoid',
])
def test_pairwise_kernels(kernel):
    X = da.random.uniform(size=(100, 4), chunks=(50, 4))
    a = dm.pairwise.PAIRWISE_KERNEL_FUNCTIONS[kernel]
    b = sm.pairwise.PAIRWISE_KERNEL_FUNCTIONS[kernel]

    r1 = a(X)
    r2 = b(X.compute())
    assert isinstance(X, da.Array)
    assert_eq(r1, r2)
