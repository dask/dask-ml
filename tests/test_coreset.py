import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from dask_ml.cluster.coreset import Coreset, get_m, lightweight_coresets
from dask_ml.datasets import make_classification
from dask_ml.utils import assert_estimator_equal


class DummyEstimator(BaseEstimator):
    pass


@pytest.mark.parametrize(
    "kwargs, m",
    [
        (dict(X=da.ones((10000, 5)), k=5, eps=0.1), 403),
        (dict(X=da.ones((10000, 10)), k=5, eps=0.01), 8048),
        (dict(X=da.ones((10000, 10)), k=5, eps=0.05), 1610),
        (dict(X=da.ones((10000, 5)), k=3, eps=0.2), 83),
        (dict(X=da.ones((10000, 2)), k=20, eps=0.2), 600),
        (dict(X=da.ones((10000, 10)), k=20, eps=0.05), 40000),  # m > 10k -> fallback
        (dict(X=da.ones((10000, 3)), k=3, eps=0.1, mode="soft"), 810),
        (dict(X=da.ones((10000, 3)), k=3, eps=0.1, mode="hard"), 99),
    ],
)
def test_get_m(kwargs, m):
    # See Theorem 2 and Section 6. from
    # https://dl.acm.org/doi/pdf/10.1145/3219819.3219973
    # `d` in the theorem is simply X.shape[1]
    computed_m = get_m(**kwargs)
    assert int(computed_m) == m


@pytest.mark.parametrize(
    "estimator, kwargs, error",
    [
        (DummyEstimator(), dict(eps=0.05, delta=0.2, m=None), ValueError),
        (
            DummyEstimator(),
            dict(m=200),
            None,
        ),  # m is here, no need for n_clusters|n_components
        (KMeans(), dict(eps=2), ValueError),  # eps between 0 and 1
        (KMeans(), dict(delta=2), ValueError),  # delta between 0 and 1
        (KMeans(), dict(eps=0.1), None),  # extracting `n_clusters` attr
        (GaussianMixture(), dict(delta=0.2), None),  # extracting `n_components` attr
        (KMeans(), dict(eps=0.02, delta=0.2, m=None), ValueError),
    ],
)
def test_init(estimator, kwargs, error):
    if error is None:
        Coreset(estimator, **kwargs)
    else:
        with pytest.raises(error):
            Coreset(estimator, **kwargs)


def test_lightweight_coresets():
    X = da.array([[3, 5], [3, 10], [4, 4]])
    gen = da.random.RandomState(3)
    Xcs, wcs = lightweight_coresets(X, 2, gen=gen)
    np.testing.assert_array_equal(Xcs.compute(), X[[2, 1]].compute())

    np.testing.assert_array_almost_equal(
        wcs, np.array([2.67948718, 0.836]), decimal=3,
    )


class TestKMeans:
    def test_basic(self, Xl_blobs_easy):
        X, _ = Xl_blobs_easy
        m = X.shape[0] / 2

        # make it super easy to cluster
        skkm = KMeans(n_clusters=3, random_state=0)
        dkkm = Coreset(KMeans(n_clusters=3, random_state=0), m=m)
        skkm.fit(X)
        dkkm.fit(X)

        assert dkkm.m == m
        assert_estimator_equal(
            skkm, dkkm, exclude=["n_iter_", "inertia_", "cluster_centers_", "labels_"]
        )

        # sampling should reduce absolute sum of squared distances
        assert dkkm.inertia_ <= skkm.inertia_

        assert dkkm.n_iter_

    @pytest.mark.parametrize("eps", [0.05, 0.2])
    @pytest.mark.parametrize("k", [3, 10])
    def test_inertia(self, eps, k):
        """
        Test we find a (eps, k)-lightweight coreset of X
        for different values of `eps` and `k`

        See section 2 from
        https://dl.acm.org/doi/pdf/10.1145/3219819.3219973
        """
        X, _ = make_classification(
            n_samples=10_000, n_features=k, chunks=100, random_state=1
        )

        def get_inertia(est, X):
            """
            The `intertia_` attribute is relative to the fitted data,
            We have to compute intertia regarding to the entire input data
            for the coreset version, if we want to compare it with the non-coreset one
            """
            return (est.transform(X).min(axis=1) ** 2).sum()

        skkm = KMeans(n_clusters=k, random_state=0)
        dkkm = Coreset(KMeans(n_clusters=k, random_state=0), eps=eps)
        skkm.fit(X)
        dkkm.fit(X)

        assert_estimator_equal(
            skkm, dkkm, exclude=["n_iter_", "inertia_", "cluster_centers_", "labels_"]
        )

        dkkm_X_inertia = get_inertia(dkkm, X).compute()

        # See section 2. formulae 2.
        assert dkkm_X_inertia <= (1 + 2 * eps) * skkm.inertia_


def test_dataframes():
    df = dd.from_pandas(
        pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}), npartitions=2
    )

    kmeans = Coreset(KMeans(n_clusters=2))
    kmeans.fit(df)
