import contextlib
import string

import dask.dataframe as dd
import numpy as np
import pytest
from distributed.utils_test import cluster

from dask_ml.datasets import (
    make_blobs,
    make_classification,
    make_counts,
    make_regression,
)

# pytest.register_assert_rewrite('dask_ml.utils')


@pytest.fixture
def xy_classification():
    """X, y pair for classification"""
    X, y = make_classification(chunks=(10, 20), random_state=0)
    return X, y


@pytest.fixture
def xy_classification_pandas(xy_classification):
    """Dask DataFrame (Series) X, y pair for classification"""
    X, y = xy_classification
    X = dd.from_dask_array(X, columns=list(string.ascii_letters[: X.shape[1]]))
    y = dd.from_dask_array(y, "target")
    return X, y


@pytest.fixture
def xy_regression():
    """X, y pair for classification"""
    X, y = make_regression(chunks=(10, 20), random_state=0)
    return X, y


@pytest.fixture
def xy_counts():
    """X, y pair for predicting counts"""
    X, y = make_counts(n_samples=100, n_features=5, chunks=10)
    return X, y


@pytest.fixture
def Xl_blobs():
    """
    Tuple of (X, labels) for a classification task. `X`
    and `l` are both dask arrays
    """
    X, l = make_classification(n_samples=1000, n_features=4, chunks=500, random_state=1)
    return X, l


@pytest.fixture
def Xl_blobs_easy():
    """
    Tuple of (X, labels) for classification.

    The centers are very spread out, so the clustering is easy.
    """
    centers = np.array([[-7, -7], [0, 0], [7, 7]])
    X, y = make_blobs(cluster_std=0.1, centers=centers, chunks=50, random_state=0)
    return X, y


@pytest.fixture
def X_blobs(Xl_blobs):
    """
    X dataset from `Xl_blobs`.
    """
    return Xl_blobs[0]


@pytest.fixture
def single_chunk_classification():
    """X, y pair for classification.

    The `X` and `y` have a single block, so chunksize is 100.
    Useful for testing `partial_fit` methods.
    """
    X, y = make_classification(chunks=100, random_state=0)
    return X, y


@pytest.fixture
def single_chunk_regression():
    """X, y pair for regression.

    The `X` and `y` have a single block, so chunksize is 100.
    Useful for testing `partial_fit` methods.
    """
    X, y = make_regression(chunks=100, random_state=0)
    return X, y


@pytest.fixture
def single_chunk_count_classification():
    """X, y pair for classification.

    The `X` and `y` have a single block, so chunksize is 100.
    Useful for testing `partial_fit` methods. The `X` data
    is count data
    """
    X, y = make_classification(chunks=100, random_state=0)
    X = (abs(X) * 10).astype(int)
    return X, y


@pytest.fixture
def single_chunk_binary_classification():
    """X, y pair for classification.

    The `X` and `y` have a single block, so chunksize is 100.
    Useful for testing `partial_fit` methods. The `X` data
    are binary features
    """
    X, y = make_classification(chunks=100, random_state=0)
    X = (abs(X) > 0).astype(int)
    return X, y


@pytest.fixture
def single_chunk_blobs():
    """X, y pair for clustering

    The `X` and `y` have a single block, so chunksize is 100.
    Useful for testing `partial_fit` methods.
    """
    X, y = make_blobs(chunks=100, random_state=0)
    return X, y


@contextlib.contextmanager
def not_cluster(nworkers=2, **kwargs):
    yield (None, [None] * nworkers)


@pytest.fixture(scope="module", params=["threads", "distributed"])
def scheduler(request):
    if request.param == "distributed":
        yield cluster
    else:
        yield not_cluster
