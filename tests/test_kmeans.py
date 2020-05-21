"""
Mostly just smoke tests, and verifying that the parallel implementation is
the same as the serial.
"""
import warnings

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import sklearn.utils.extmath
from dask.array.utils import assert_eq
from sklearn.cluster import KMeans as SKKMeans
from sklearn.utils.estimator_checks import check_estimator

import dask_ml.cluster
from dask_ml.cluster import KMeans as DKKMeans, k_means
from dask_ml.cluster._compat import _k_init
from dask_ml.utils import assert_estimator_equal, row_norms


def test_check_estimator():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore", RuntimeWarning)
        check_estimator(DKKMeans())


def test_row_norms(X_blobs):
    result = row_norms(X_blobs, squared=True)
    expected = sklearn.utils.extmath.row_norms(X_blobs.compute(), squared=True)
    assert_eq(result, expected)


def replace(a, old, new):
    arr = np.empty(a.max() + 1, dtype=new.dtype)
    arr[old] = new
    return arr[a]


def test_too_small():
    km = DKKMeans()
    X = da.random.uniform(size=(20, 2), chunks=(10, 2))
    km.fit(X)


def test_fit_raises():
    km = DKKMeans()
    with pytest.raises(ValueError):
        km.fit(np.array([]).reshape(0, 1))

    with pytest.raises(ValueError):
        km.fit(np.array([]).reshape(1, 0))


class TestKMeans:
    def test_basic(self, Xl_blobs_easy):
        X, _ = Xl_blobs_easy

        # make it super easy to cluster
        a = DKKMeans(n_clusters=3, random_state=0)
        b = SKKMeans(n_clusters=3, random_state=0)
        a.fit(X)
        b.fit(X)
        assert_estimator_equal(
            a, b, exclude=["n_iter_", "inertia_", "cluster_centers_", "labels_"]
        )
        assert abs(a.inertia_ - b.inertia_) < 0.01
        # order is arbitrary, so align first
        a_order = np.argsort(a.cluster_centers_, 0)[:, 0]
        b_order = np.argsort(b.cluster_centers_, 0)[:, 0]
        a_centers = a.cluster_centers_[a_order]
        b_centers = b.cluster_centers_[b_order]
        np.testing.assert_allclose(a_centers, b_centers, rtol=1e-3)
        b_labels = replace(b.labels_, [0, 1, 2], a_order[b_order]).astype(
            b.labels_.dtype
        )
        assert_eq(a.labels_.compute(), b_labels)
        assert a.n_iter_
        # this is hacky
        b.cluster_centers_ = b_centers
        a.cluster_centers_ = a_centers
        assert_eq(a.transform(X), b.transform(X), rtol=1e-3)

        yhat_a = a.predict(X)
        yhat_b = b.predict(X)
        assert_eq(yhat_a.compute(), yhat_b)

    def test_fit_given_init(self):
        X, y = sklearn.datasets.make_blobs(n_samples=1000, n_features=4, random_state=1)
        X = da.from_array(X, chunks=500)
        X_ = X.compute()
        x_squared_norms = sklearn.utils.extmath.row_norms(X_, squared=True)
        rs = np.random.RandomState(0)
        init = _k_init(X_, 3, x_squared_norms, rs)
        dkkm = DKKMeans(3, init=init, random_state=0)
        skkm = SKKMeans(3, init=init, random_state=0, n_init=1)
        dkkm.fit(X)
        skkm.fit(X_)
        assert abs(skkm.inertia_ - dkkm.inertia_) < 0.001

    def test_kmeanspp_init(self, Xl_blobs_easy):
        X, y = Xl_blobs_easy
        X_ = X.compute()
        rs = np.random.RandomState(0)
        dkkm = DKKMeans(3, init="k-means++", random_state=rs)
        skkm = SKKMeans(3, init="k-means++", random_state=rs)
        dkkm.fit(X)
        skkm.fit(X_)
        assert abs(dkkm.inertia_ - skkm.inertia_) < 1e-4
        assert dkkm.init == "k-means++"

    def test_kmeanspp_init_random_state(self, Xl_blobs_easy):
        X, y = Xl_blobs_easy
        a = DKKMeans(3, init="k-means++")
        a.fit(X)

        b = DKKMeans(3, init="k-means++", random_state=0)
        b.fit(X)

    @pytest.mark.xfail(reason="Flaky")
    def test_random_init(self, Xl_blobs_easy):
        X, y = Xl_blobs_easy
        X_ = X.compute()
        rs = 0
        dkkm = DKKMeans(3, init="random", random_state=rs)
        skkm = SKKMeans(3, init="random", random_state=rs, n_init=1)
        dkkm.fit(X)
        skkm.fit(X_)
        assert abs(dkkm.inertia_ - skkm.inertia_) < 1e-4
        assert dkkm.init == "random"

    def test_invalid_init(self, Xl_blobs_easy):
        X, y = Xl_blobs_easy
        init = X[:2].compute()

        with pytest.raises(ValueError):
            k_means.k_init(X, 3, init)

        init = X[:2, :-1].compute()

        with pytest.raises(ValueError):
            k_means.k_init(X, 2, init)

        with pytest.raises(ValueError):
            k_means.k_init(X, 2, "invalid")

        with pytest.raises(TypeError):
            k_means.k_init(X, 2, 2)

    @pytest.mark.parametrize(
        "X",
        [
            np.random.uniform(size=(100, 4)),
            da.random.uniform(size=(100, 4), chunks=(10, 4)),
            pd.DataFrame(np.random.uniform(size=(100, 4))),
        ],
    )
    def test_inputs(self, X):
        km = DKKMeans(n_clusters=3)
        km.fit(X)
        km.transform(X)

    def test_dtypes(self):
        X = da.random.uniform(size=(100, 2), chunks=(50, 2))
        X2 = X.astype("f4")
        pairs = [(X, X), (X2, X2), (X, X2), (X2, X)]

        for xx, yy in pairs:
            a = DKKMeans()
            b = SKKMeans()
            a.fit(xx)
            b.fit(xx)
            assert a.cluster_centers_.dtype == b.cluster_centers_.dtype
            assert a.labels_.dtype == b.labels_.dtype
            assert a.transform(xx).dtype == b.transform(xx).dtype
            assert a.transform(yy).dtype == b.transform(yy).dtype


def test_dataframes():
    df = dd.from_pandas(
        pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}), npartitions=2
    )

    kmeans = DKKMeans()
    kmeans.fit(df)


def test_empty_chunks():
    # https://github.com/dask/dask-ml/issues/551
    X = da.random.random((100, 4), chunks=((0, 5, 95), (4,)))
    trn = dask_ml.cluster.KMeans(n_clusters=2).fit_transform(X)  # it works
    assert trn.chunks == ((5, 95), (2,))
