import pytest
import sklearn.cluster

from dask_ml.datasets import make_blobs
from dask_ml.cluster import SpectralClustering


X, y = make_blobs(n_samples=100, chunks=50)
X_ = X.compute()


@pytest.mark.parametrize('data', [X, X_])
def test_basic(data):
    sc = SpectralClustering(n_components=25, random_state=0)
    sc.fit(data)


def test_sklearn_kmeans():
    km = sklearn.cluster.KMeans(n_init=2)
    sc = SpectralClustering(n_components=25, random_state=0, assign_labels=km)
    sc.fit(X)
