"""
Mostly just smoke tests, and verifying that the parallel implementation is
the same as the serial.
"""
import numpy as np
from dask.array.utils import assert_eq
from daskml.cluster import KMeans as DKKMeans
from daskml.cluster import k_means
from daskml.utils import assert_estimator_equal
from sklearn.cluster import KMeans as SKKMeans
from sklearn.cluster import k_means_


def test_row_norms(X_blobs):
    result = k_means.row_norms(X_blobs, squared=True)
    expected = k_means_.row_norms(X_blobs.compute(), squared=True)
    assert_eq(result, expected)


def replace(a, old, new):
    arr = np.empty(a.max()+1, dtype=new.dtype)
    arr[old] = new
    return arr[a]


class TestKMeans:

    def test_basic(self, Xl_blobs_easy):
        X, _ = Xl_blobs_easy

        # make it super easy to cluster
        a = DKKMeans(n_clusters=3, random_state=0)
        b = SKKMeans(n_clusters=3, random_state=0)
        a.fit(X)
        b.fit(X)
        assert_estimator_equal(a, b, exclude=['n_iter_', 'inertia_',
                                              'cluster_centers_',
                                              'labels_'])
        assert abs(a.inertia_ - b.inertia_) < 0.01
        # order is arbitrary, so align first
        a_order = np.argsort(a.cluster_centers_, 0)[:, 0]
        b_order = np.argsort(b.cluster_centers_, 0)[:, 0]
        a_centers = a.cluster_centers_[a_order]
        b_centers = b.cluster_centers_[b_order]
        np.testing.assert_allclose(a_centers, b_centers,
                                   rtol=1e-3)
        b_labels = replace(b.labels_, [0, 1, 2], a_order[b_order])
        assert_eq(a.labels_.compute(), b_labels)
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
        assert_eq(dkkm.inertia_, skkm.inertia_)

    def test_kmeanspp_init(self, Xl_blobs_easy):
        X, y = Xl_blobs_easy
        X_ = X.compute()
        rs = np.random.RandomState(0)
        dkkm = DKKMeans(3, init='k-means++', random_state=rs)
        skkm = SKKMeans(3, init='k-means++', random_state=rs, n_init=1)
        dkkm.fit(X)
        skkm.fit(X_)
        assert abs(dkkm.inertia_ - skkm.inertia_) < 1e-4
        assert dkkm.init == 'k-means++'
