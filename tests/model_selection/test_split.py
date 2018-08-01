import dask.array as da
import numpy as np
import pandas as pd
import pytest
import six
from sklearn.datasets import fetch_20newsgroups, make_regression

import dask_ml.model_selection

X, y = make_regression(n_samples=110, n_features=5)
dX = da.from_array(X, 50)
dy = da.from_array(y, 50)


def test_20_newsgroups():
    data = fetch_20newsgroups()
    X, y = data.data, data.target
    r = dask_ml.model_selection.train_test_split(X, y)
    X_train, X_test, y_train, y_test = r
    for X in [X_train, X_test]:
        assert isinstance(X, list)
        assert isinstance(X[0], six.string_types)
    for y in [y_train, y_test]:
        assert isinstance(y, np.ndarray)
        assert y.dtype == int


def test_blockwise_shufflesplit():
    splitter = dask_ml.model_selection.ShuffleSplit(random_state=0)
    assert splitter.get_n_splits() == 10
    gen = splitter.split(dX)

    train_idx, test_idx = next(gen)
    assert isinstance(train_idx, da.Array)
    assert isinstance(test_idx, da.Array)

    assert train_idx.shape == (99,)  # 90% of 110
    assert test_idx.shape == (11,)

    assert train_idx.chunks == ((45, 45, 9),)
    assert test_idx.chunks == ((5, 5, 1),)

    counts = pd.value_counts(train_idx.compute())
    assert counts.max() == 1

    N = len(X)

    np.testing.assert_array_equal(
        np.unique(da.concatenate([train_idx, test_idx])), np.arange(N)
    )


def test_train_test_split():
    X_train, X_test, y_train, y_test = dask_ml.model_selection.train_test_split(
        dX, dy, random_state=10
    )

    assert len(X_train) == 99
    assert len(X_test) == 11

    assert X_train.chunks[0] == y_train.chunks[0]
    assert X_test.chunks[0] == y_test.chunks[0]


def test_train_test_split_test_size():
    X_train, X_test, y_train, y_test = dask_ml.model_selection.train_test_split(
        dX, dy, random_state=10, test_size=0.8
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"train_size": 10}, {"test_size": 10}, {"test_size": 10, "train_size": 0.1}],
)
def test_absolute_raises(kwargs):
    with pytest.raises(ValueError) as m:
        dask_ml.model_selection.train_test_split(dX, **kwargs)
    assert m.match("Dask-ML does not support absolute sizes")


def test_non_complement_raises():
    with pytest.raises(ValueError) as m:
        dask_ml.model_selection._split._maybe_normalize_split_sizes(0.1, 0.2)
    assert m.match("The sum of ")


def test_complement():
    train_size, test_size = dask_ml.model_selection._split._maybe_normalize_split_sizes(
        0.1, None
    )
    assert train_size == 0.1
    assert test_size == 0.9

    train_size, test_size = dask_ml.model_selection._split._maybe_normalize_split_sizes(
        None, 0.2
    )
    assert train_size == 0.8
    assert test_size == 0.2
