import dask.array as da
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from dask_ml.cluster.coreset import Coreset, lightweight_coresets


class DummyEstimator(BaseEstimator):
    pass


@pytest.mark.parametrize(
    "est, m, eps, error",
    [
        (DummyEstimator(), 200, 0.05, None),
        (DummyEstimator(), None, 0.05, ValueError),
        (KMeans(), None, 0.3, None),  # Kmeans does have a `n_clusters` attribute
        (
            GaussianMixture(),
            None,
            0.3,
            None,
        ),  # GaussianMixture has a `n_components` attribute
    ],
)
def test_init(est, m, eps, error):
    if error is None:
        Coreset(est, m, eps=eps)
    else:
        with pytest.raises(error):
            Coreset(est, m, eps=eps)


def test_lighweight_coresets():
    X = da.array([[3, 5], [3, 10], [4, 4]])
    gen = da.random.RandomState(3)
    Xcs, wcs = lightweight_coresets(X, 2, gen=gen)
    np.testing.assert_array_equal(Xcs.compute(), X[[2, 1]].compute())

    np.testing.assert_array_almost_equal(
        wcs, np.array([2.67948718, 0.836]), decimal=3,
    )
