from functools import partial

import pytest
import sklearn.cluster
import numpy as np
from numpy.testing import assert_array_equal

from dask_ml.datasets import make_blobs
from dask_ml.cluster import SpectralClustering
from dask_ml import metrics


X, y = make_blobs(n_samples=200, chunks=100, random_state=0)


@pytest.mark.parametrize('as_ndarray', [False, True])
@pytest.mark.parametrize('persist_embedding', [True, False])
def test_basic(as_ndarray, persist_embedding):
    sc = SpectralClustering(n_components=25, random_state=0,
                            persist_embedding=persist_embedding)
    sc.fit(X.compute())
    assert len(sc.labels_) == len(X)


@pytest.mark.parametrize('assign_labels', [
    sklearn.cluster.KMeans(n_init=2),
    'sklearn-kmeans'])
def test_sklearn_kmeans(assign_labels):
    sc = SpectralClustering(n_components=25, random_state=0,
                            assign_labels=assign_labels,
                            kmeans_params={'n_clusters': 8})
    sc.fit(X)
    assert isinstance(sc.assign_labels_, sklearn.cluster.KMeans)


def test_callable_affinity():
    affinity = partial(metrics.pairwise.pairwise_kernels,
                       metric='rbf',
                       filter_params=True)
    sc = SpectralClustering(affinity=affinity)
    sc.fit(X)


def test_n_components_raises():
    sc = SpectralClustering(n_components=len(X))
    with pytest.raises(ValueError) as m:
        sc.fit(X)
    assert m.match('n_components')


def test_assign_labels_raises():
    sc = SpectralClustering(assign_labels='foo')
    with pytest.raises(ValueError) as m:
        sc.fit(X)

    assert m.match("Unknown 'assign_labels' 'foo'")

    sc = SpectralClustering(assign_labels=dict())
    with pytest.raises(TypeError) as m:
        sc.fit(X)

    assert m.match("Invalid type ")


def test_affinity_raises():
    sc = SpectralClustering(affinity='foo')
    with pytest.raises(ValueError) as m:
        sc.fit(X)

    assert m.match("Unknown affinity metric name 'foo'")

    sc = SpectralClustering(affinity=np.array([]))
    with pytest.raises(TypeError) as m:
        sc.fit(X)
        assert m.match("Unexpected type for affinity 'ndarray'")


def test_spectral_clustering():
    S = np.array([[1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0],
                  [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

    model = SpectralClustering(random_state=0, n_clusters=2,
                               n_components=4).fit(S)
    labels = model.labels_.compute()
    if labels[0] == 0:
        labels = 1 - labels

    assert_array_equal(labels, [1, 1, 1, 0, 0, 0, 0])
