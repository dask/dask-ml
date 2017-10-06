import pytest
from daskml.datasets import (make_classification,
                             make_regression,
                             make_blobs,
                             make_counts)


@pytest.fixture
def xy_classification():
    """X, y pair for classification"""
    X, y = make_classification(chunks=10, random_state=0)
    return X, y


@pytest.fixture
def xy_regression():
    """X, y pair for classification"""
    X, y = make_regression(chunks=10, random_state=0)
    return X, y


@pytest.fixture
def xy_counts():
    """X, y pair for predicting counts"""
    X, y = make_counts(n_samples=100, n_features=5, chunks=10)
    return X, y


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
