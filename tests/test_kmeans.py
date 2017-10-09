"""
Mostly just smoke tests, and verifying that the parallel implementation is
the same as the serial.
"""
import numpy as np
from dask.array.utils import assert_eq
from sklearn.cluster import k_means_
from sklearn.cluster import KMeans as SKKMeans

from daskml.cluster import k_means
from daskml.cluster import KMeans as DKKMeans

from daskml.utils import assert_estimator_equal


def test_row_norms(X_blobs):
    result = k_means.row_norms(X_blobs, squared=True)
    expected = k_means_.row_norms(X_blobs.compute(), squared=True)
    assert_eq(result, expected)


class TestKMeans:

    def test_basic(self, X_blobs):
        a = DKKMeans()
        b = SKKMeans()
        a.fit(X_blobs)
        b.fit(X_blobs)
        assert_estimator_equal(a, b)

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
