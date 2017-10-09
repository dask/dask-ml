from sklearn import metrics as metrics_
from dask.array.utils import assert_eq
import numpy.testing as npt

from daskml import metrics


def test_pairwise_distances(X_blobs):
    centers = X_blobs[::100].compute()
    result = metrics.pairwise_distances(X_blobs, centers)
    expected = metrics_.pairwise_distances(X_blobs.compute(), centers)
    assert_eq(result, expected)


def test_pairwise_distances_argmin_min(X_blobs):
    centers = X_blobs[::100].compute()
    a_, b_ = metrics_.pairwise_distances_argmin_min(X_blobs.compute(), centers)
    a, b = metrics.pairwise_distances_argmin_min(X_blobs, centers)
    npt.assert_array_equal(a.compute(), a_)
    npt.assert_array_equal(b.compute(), b_)
