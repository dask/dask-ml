import pytest
import sklearn.cluster
import numpy as np
from numpy.testing import assert_array_equal

from dask_ml.datasets import make_blobs
from dask_ml.cluster import SpectralClustering


X, y = make_blobs(n_samples=100, chunks=50, random_state=0)
X_ = X.compute()


@pytest.mark.parametrize('data', [X, X_])
def test_basic(data):
    sc = SpectralClustering(n_components=25, random_state=0)
    sc.fit(data)
    assert len(sc.labels_) == len(X)


def test_sklearn_kmeans():
    km = sklearn.cluster.KMeans(n_init=2)
    sc = SpectralClustering(n_components=25, random_state=0, assign_labels=km)
    sc.fit(X)


def test_spectral_clustering():
    S = np.array([[1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
                  [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

    model = SpectralClustering(random_state=0, n_clusters=2).fit(S)
    labels = model.labels_.compute()
    if labels[0] == 0:
        labels = 1 - labels

    assert_array_equal(labels, [1, 1, 1, 0, 0, 0, 0])
