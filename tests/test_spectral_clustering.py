from functools import partial

import numpy as np
import pytest
import sklearn.cluster

from dask_ml import metrics
from dask_ml.cluster import SpectralClustering
from dask_ml.datasets import make_blobs

X, y = make_blobs(n_samples=200, chunks=100, random_state=0)


@pytest.mark.parametrize("as_ndarray", [False, True])
@pytest.mark.parametrize("persist_embedding", [True, False])
def test_basic(as_ndarray, persist_embedding):
    sc = SpectralClustering(
        n_components=25, random_state=0, persist_embedding=persist_embedding
    )
    if as_ndarray:
        X_ = X.compute()
    else:
        X_ = X
    sc.fit(X_)
    assert len(sc.labels_) == len(X_)


@pytest.mark.parametrize(
    "assign_labels", [sklearn.cluster.KMeans(n_init=2), "sklearn-kmeans"]
)
def test_sklearn_kmeans(assign_labels):
    sc = SpectralClustering(
        n_components=25,
        random_state=0,
        assign_labels=assign_labels,
        kmeans_params={"n_clusters": 8},
    )
    sc.fit(X)
    assert isinstance(sc.assign_labels_, sklearn.cluster.KMeans)


@pytest.mark.skip(reason="Can't reproduce CI failure.")
def test_callable_affinity():
    affinity = partial(
        metrics.pairwise.pairwise_kernels,
        metric="rbf",
        filter_params=True,
        gamma=1.0 / len(X),
    )
    sc = SpectralClustering(affinity=affinity, gamma=None)
    sc.fit(X)


def test_n_components_raises():
    sc = SpectralClustering(n_components=len(X))
    with pytest.raises(ValueError) as m:
        sc.fit(X)
    assert m.match("n_components")


def test_assign_labels_raises():
    sc = SpectralClustering(assign_labels="foo")
    with pytest.raises(ValueError) as m:
        sc.fit(X)

    assert m.match("Unknown 'assign_labels' 'foo'")

    sc = SpectralClustering(assign_labels=dict())
    with pytest.raises(TypeError) as m:
        sc.fit(X)

    assert m.match("Invalid type ")


def test_affinity_raises():
    sc = SpectralClustering(affinity="foo")
    with pytest.raises(ValueError) as m:
        sc.fit(X)

    assert m.match("Unknown affinity metric name 'foo'")

    sc = SpectralClustering(affinity=np.array([]))
    with pytest.raises(TypeError) as m:
        sc.fit(X)
        assert m.match("Unexpected type for affinity 'ndarray'")


def test_spectral_clustering(Xl_blobs_easy):
    X, y = Xl_blobs_easy
    X = (X - X.mean(0)) / X.std(0)
    model = SpectralClustering(
        random_state=0, n_clusters=3, n_components=5, gamma=None
    ).fit(X)
    labels = model.labels_.compute()
    y = y.compute()

    idx = [(y == i).argmax() for i in range(3)]
    grouped_idx = [np.where(y == y[idx[i]])[0] for i in range(3)]

    for indices in grouped_idx:
        assert len(set(labels[indices])) == 1


@pytest.mark.parametrize("keep", [[4, 7], [4, 5], [0, 3], [1, 9], [0, 1, 5, 8, 9]])
def test_slice_mostly_sorted(keep):
    import numpy as np
    import dask.array as da
    from dask.array.utils import assert_eq
    from dask_ml.cluster.spectral import _slice_mostly_sorted

    X = np.arange(10).reshape(-1, 1)

    dX = da.from_array(X, chunks=5)
    keep = np.array(keep).ravel()
    rest = ~np.isin(X, keep).ravel()

    array = dX[np.concatenate([X[keep], X[rest]]).ravel()]
    result = _slice_mostly_sorted(array, keep, rest)

    assert_eq(result, X)
    assert all(x > 0 for x in result.chunks[0])
