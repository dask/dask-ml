from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
import dask
import dask.array as da
import dask.bag as db
from scipy import sparse

import dklearn.matrix as dm
from dklearn.cross_validation import random_split, RandomSplit


def test_random_split_errors():
    b = db.range(1000, npartitions=10)
    with pytest.raises(ValueError):
        random_split(b, 2)
    with pytest.raises(ValueError):
        random_split(b, -1)
    with pytest.raises(ValueError):
        random_split(b, 0.5, "not-a-seed-or-RandomState")
    with pytest.raises(TypeError):
        random_split("not-a-dask-object", 0.5)


def test_random_split_bag():
    b = db.range(1000, npartitions=10)
    train, test = random_split(b, 0.2, 123)

    assert random_split(b, 0.2, 123)[0].name == train.name
    assert random_split(b, 0.3, 123)[0].name != train.name
    assert random_split(b, 0.2)[0].name != random_split(b, 0.2)[0].name

    train_c, test_c = dask.compute(train, test)
    assert 0.75 < len(train_c) / 1000 < 0.85
    assert len(train_c) + len(test_c) == 1000
    assert set(train_c) | set(test_c) == set(range(1000))


def test_random_split_matrix():
    a = np.arange(1000)
    m = dm.from_array(da.from_array(a, chunks=100))
    train, test = random_split(m, 0.2, 123)
    assert train.dtype == test.dtype == m.dtype
    assert train.ndim == test.ndim == 1
    assert train.shape == test.shape == (None,)

    assert random_split(m, 0.2, 123)[0].name == train.name
    assert random_split(m, 0.3, 123)[0].name != train.name
    assert random_split(m, 0.2)[0].name != random_split(m, 0.2)[0].name

    train_c, test_c = dask.compute(train, test)
    assert 0.75 < len(train_c) / 1000 < 0.85
    assert len(train_c) + len(test_c) == 1000
    assert set(train_c) | set(test_c) == set(range(1000))

    # 2D
    a = np.arange(1000).reshape((1000, 1))
    m = dm.from_array(da.from_array(a, chunks=100))
    train, test = random_split(m, 0.2, 123)
    assert train.dtype == test.dtype == m.dtype
    assert train.ndim == test.ndim == 2
    assert train.shape == test.shape == (None, 1)

    train_c, test_c = dask.compute(train, test)
    assert 0.75 < len(train_c) / 1000 < 0.85
    assert len(train_c) + len(test_c) == 1000

    # Sparse
    m = m.map_partitions(sparse.csr_matrix, shape=m.shape, dtype=m.dtype)
    train, test = random_split(m, 0.2, 123)
    train_c, test_c = dask.compute(train, test)
    assert 0.75 < train_c.shape[0] / 1000 < 0.85
    assert train_c.shape[0] + test_c.shape[0] == 1000
    assert sparse.issparse(train_c) and sparse.issparse(test_c)


def test_random_split_array():
    a = np.arange(1000)
    x = da.from_array(a, chunks=100)
    train, test = random_split(x, 0.2, 123)
    assert train.dtype == test.dtype == x.dtype

    assert random_split(x, 0.2, 123)[0].name == train.name
    assert random_split(x, 0.3, 123)[0].name != train.name
    assert random_split(x, 0.2)[0].name != random_split(x, 0.2)[0].name

    train_c, test_c = dask.compute(train, test)
    assert train_c.shape == train.shape
    assert test_c.shape == test.shape
    assert 0.75 < len(train_c) / 1000 < 0.85
    assert len(train_c) + len(test_c) == 1000
    assert set(train_c) | set(test_c) == set(range(1000))

    # 2D
    a = np.arange(1000).reshape((1000, 1))
    x = da.from_array(a, chunks=100)
    train, test = random_split(x, 0.2, 123)
    assert train.dtype == test.dtype == x.dtype

    train_c, test_c = dask.compute(train, test)
    assert train_c.shape == train.shape
    assert test_c.shape == test.shape
    assert 0.75 < len(train_c) / 1000 < 0.85
    assert len(train_c) + len(test_c) == 1000


def test_RandomSplit():
    X = np.arange(10000).reshape((1000, 10))
    y = np.arange(1000)
    dX = da.from_array(X, chunks=(100, 5))
    dy = da.from_array(y, chunks=100)

    rs = RandomSplit(n_iter=3, test_size=0.2, random_state=123)
    sets = list(rs.split(dX, dy))
    assert len(sets) == 3
    assert len(sets[0]) == 4
    X_train, y_train, X_test, y_test = sets[0]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(X)

    rs2 = RandomSplit(n_iter=3, test_size=0.2, random_state=123)
    sets2 = list(rs2.split(dX, dy))
    for s1, s2 in zip(sets, sets2):
        for a1, a2 in zip(s1, s2):
            assert a1.name == a2.name

    sets3 = list(rs2.split(dX))
    assert len(sets3[0]) == 2
    X_train, X_test = sets3[0]
    assert len(X_train) + len(X_test) == len(X)
