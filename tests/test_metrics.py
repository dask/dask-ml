import dask.array as da
import numpy as np
import numpy.testing as npt
from dask.array.utils import assert_eq
import sklearn.metrics as sm

import daskml.metrics as dm


def test_pairwise_distances(X_blobs):
    centers = X_blobs[::100].compute()
    result = dm.pairwise_distances(X_blobs, centers)
    expected = sm.pairwise_distances(X_blobs.compute(), centers)
    assert_eq(result, expected)


def test_pairwise_distances_argmin_min(X_blobs):
    centers = X_blobs[::100].compute()
    a_, b_ = sm.pairwise_distances_argmin_min(X_blobs.compute(), centers)
    a, b = dm.pairwise_distances_argmin_min(X_blobs, centers)
    npt.assert_array_equal(a.compute(), a_)
    npt.assert_array_equal(b.compute(), b_)


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
    assert_eq(a, b)

    x_norm_squared = (X ** 2).sum(axis=1).compute()[:, np.newaxis]
    assert_eq(X, X, Y_norm_squared=x_norm_squared, atol=1e-5)
