"""
Mostly just smoke tests, and verifying that the parallel implementation is
the same as the serial.
"""
import numpy as np
from dask.array.utils import assert_eq
from daskml.cluster import KMeans as DKKMeans
from daskml.cluster import k_means
from daskml.datasets import make_blobs
from daskml.utils import assert_estimator_equal
from sklearn.cluster import KMeans as SKKMeans
from sklearn.cluster import k_means_


def test_row_norms(X_blobs):
    result = k_means.row_norms(X_blobs, squared=True)
    expected = k_means_.row_norms(X_blobs.compute(), squared=True)
    assert_eq(result, expected)


class TestKMeans:

    def test_basic(self):

        # make it super easy to cluster
        centers = np.array([
            [-7, -7],
            [0, 0],
            [7, 7],
        ])
        X, y = make_blobs(cluster_std=0.1, centers=centers, chunks=50,
                          random_state=0)
        a = DKKMeans(random_state=0)
        b = SKKMeans(random_state=0)
        a.fit(X)
        b.fit(X)
        assert_estimator_equal(a, b, exclude=['n_iter_', 'inertia_',
                                              'cluster_centers_'])
        assert abs(a.inertia_ - b.inertia_) < 0.01
        np.testing.assert_allclose(np.sort(a.cluster_centers_, 0),
                                   np.sort(b.cluster_centers_, 0))

        assert a.n_iter_

    def test_fit_given_init(self, X_blobs):
        X_ = X_blobs.compute()
        x_squared_norms = k_means_.row_norms(X_, squared=True)
        rs = np.random.RandomState(0)
        init = k_means_._k_init(X_, 3, x_squared_norms, rs)
        dkkm = DKKMeans(3, init=init, random_state=rs)
        skkm = SKKMeans(3, init=init, random_state=rs, n_init=1)
        dkkm.fit(X_blobs)
        skkm.fit(X_)
        assert_eq(dkkm.inertia_.compute(), skkm.inertia_)
