import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_20newsgroups, make_regression

import dask_ml.model_selection
from dask_ml._compat import DASK_2130

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
        assert isinstance(X[0], str)
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


def test_blockwise_shufflesplit_rng():
    # Regression test for issue #380
    n_splits = 2
    splitter = dask_ml.model_selection.ShuffleSplit(n_splits=n_splits, random_state=0)
    gen = splitter.split(dX)

    train_indices = []
    test_indices = []
    for train_idx, test_idx in gen:
        train_indices.append(train_idx)
        test_indices.append(test_idx)

    assert not np.array_equal(train_indices[0], train_indices[1])
    assert not np.array_equal(test_indices[0], test_indices[1])

    # Test that splitting is reproducible
    n_splits = 2
    split1 = dask_ml.model_selection.ShuffleSplit(n_splits=n_splits, random_state=0)
    split2 = dask_ml.model_selection.ShuffleSplit(n_splits=n_splits, random_state=0)

    for (train_1, test_1), (train_2, test_2) in zip(split1.split(dX), split2.split(dX)):
        da.utils.assert_eq(train_1, train_2)
        da.utils.assert_eq(test_1, test_2)


@pytest.mark.parametrize("shuffle", [False, True])
def test_kfold(shuffle):
    splitter = dask_ml.model_selection.KFold(
        n_splits=5, random_state=0, shuffle=shuffle
    )
    assert splitter.get_n_splits() == 5
    gen = splitter.split(dX)

    train_idx, test_idx = next(gen)
    assert isinstance(train_idx, da.Array)
    assert isinstance(test_idx, da.Array)

    assert train_idx.shape == (88,)  # 80% of 110
    assert test_idx.shape == (22,)

    assert train_idx.chunks == ((28, 50, 10),)
    assert test_idx.chunks == ((22,),)

    counts = pd.value_counts(train_idx.compute())
    assert counts.max() == 1

    N = len(X)

    np.testing.assert_array_equal(
        np.unique(da.concatenate([train_idx, test_idx])), np.arange(N)
    )

    expected_chunks = [
        (((22, 6, 50, 10),), ((22,),)),
        (((44, 34, 10),), ((6, 16),)),
        (((50, 16, 12, 10),), ((22,),)),
        (((50, 38),), ((12, 10),)),
    ]

    for (exp_train_idx, exp_test_idx), (train_idx, test_idx) in zip(
        expected_chunks, gen
    ):
        assert train_idx.chunks == exp_train_idx
        assert test_idx.chunks == exp_test_idx


def test_train_test_split():
    X_train, X_test, y_train, y_test = dask_ml.model_selection.train_test_split(dX, dy)

    assert len(X_train) == 99
    assert len(X_test) == 11

    assert X_train.chunks[0] == y_train.chunks[0]
    assert X_test.chunks[0] == y_test.chunks[0]


def test_train_test_split_test_size():
    X_train, X_test, y_train, y_test = dask_ml.model_selection.train_test_split(
        dX, dy, random_state=10, test_size=0.8
    )


def test_train_test_split_shuffle_array():
    with pytest.raises(NotImplementedError):
        dask_ml.model_selection.train_test_split(dX, dy, shuffle=False)


@pytest.mark.xfail(
    not DASK_2130, reason="DataFrame blockwise shuffling implemented in dask2.13.0."
)
def test_train_test_split_shuffle_dataframe(xy_classification_pandas):
    X, y = xy_classification_pandas
    X_train, X_test, y_train, y_test = dask_ml.model_selection.train_test_split(
        X, y, random_state=42, shuffle=True
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(X_train.index, sorted(X_train.index))

    X_train, X_test, y_train, y_test = dask_ml.model_selection.train_test_split(
        X, y, random_state=42, shuffle=False
    )
    np.testing.assert_array_equal(X_train.index, sorted(X_train.index))


def test_train_test_split_blockwise_dataframe(xy_classification_pandas):
    X, y = xy_classification_pandas
    with pytest.raises(NotImplementedError):
        dask_ml.model_selection.train_test_split(
            X, y, random_state=42, shuffle=False, blockwise=False
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


@pytest.mark.parametrize(
    "train_size, test_size", [(None, None), (0.9, None), (None, 0.1), (0.9, 0.1)]
)
def test_train_test_split_dask_dataframe(
    xy_classification_pandas, train_size, test_size
):
    X, y = xy_classification_pandas
    kwargs = {"shuffle": True} if DASK_2130 else {}

    X_train, X_test, y_train, y_test = dask_ml.model_selection.train_test_split(
        X, y, train_size=train_size, test_size=test_size, **kwargs
    )
    assert isinstance(X_train, dd.DataFrame)
    assert isinstance(y_train, dd.Series)

    assert (y_train.size + y_test.size).compute() == len(y)


def test_train_test_split_dask_dataframe_rng(xy_classification_pandas):
    X, y = xy_classification_pandas
    kwargs = {"shuffle": True} if DASK_2130 else {}

    split1 = dask_ml.model_selection.train_test_split(
        X, y, train_size=0.25, test_size=0.75, random_state=0, **kwargs
    )

    split2 = dask_ml.model_selection.train_test_split(
        X, y, train_size=0.25, test_size=0.75, random_state=0, **kwargs
    )
    for a, b in zip(split1, split2):
        dd.utils.assert_eq(a, b)


def test_split_mixed():
    y_series = dd.from_dask_array(dy)

    with pytest.raises(TypeError, match="convert_mixed_types"):
        dask_ml.model_selection.train_test_split(dX, y_series)

    expected = dask_ml.model_selection.train_test_split(dX, dy, random_state=0)
    results = dask_ml.model_selection.train_test_split(
        dX, y_series, random_state=0, convert_mixed_types=True
    )

    assert len(expected) == len(results)
    for a, b in zip(expected, results):
        da.utils.assert_eq(a, b)


def test_split_3d_data():
    X_3d = np.arange(1.0, 5001.0).reshape((100, 10, 5))
    y_3d = np.arange(1.0, 101.0).reshape(100, 1)

    r = dask_ml.model_selection.train_test_split(X_3d, y_3d)
    X_train, X_test, y_train, y_test = r

    assert X_train.ndim == X_3d.ndim
    assert X_train.shape[1:] == X_3d.shape[1:]
