from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
import dask
import dask.array as da
import dask.bag as db
from scipy import sparse
from toolz import first
from sklearn.cross_validation import LeavePOut, StratifiedKFold

import dklearn.matrix as dm
from dklearn.cross_validation import (random_split, RandomSplit, KFold,
                                      _DaskCVWrapper, check_cv,
                                      train_test_split)


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
    X_train, y_train, X_test, y_test = sets3[0]
    assert len(X_train) + len(X_test) == len(X)
    assert y_train is None and y_test is None


def test_KFold_bag():
    X = db.range(1000, npartitions=10)
    y = db.range(1000, npartitions=10)
    cv = list(KFold(4).split(X, y))

    for x_train, y_train, x_test, y_test in cv:
        train, test = dask.compute(x_train, x_test)
        assert len(train) + len(test) == 1000
        assert set(train) | set(test) == set(range(1000))

    assert (first(KFold(3).split(X, y))[0].name ==
            first(KFold(3).split(X, y))[0].name)
    assert (first(KFold(4).split(X, y))[0].name !=
            first(KFold(3).split(X, y))[0].name)

    with pytest.raises(ValueError):
        list(KFold(11).split(X, y))


def test_Kfold_matrix():
    a = np.arange(1000)
    X = dm.from_array(da.from_array(a[:, None], chunks=100))
    y = dm.from_array(da.from_array(a.astype('f8'), chunks=100))
    cv = list(KFold(4).split(X, y))

    for x_train, y_train, x_test, y_test in cv:
        assert x_train.dtype == x_test.dtype == X.dtype
        assert x_train.shape == x_test.shape == (None, 1)
        assert x_train.ndim == x_test.ndim == 2
        assert y_train.dtype == y_test.dtype == y.dtype
        assert y_train.shape == y_test.shape == (None,)
        assert y_train.ndim == y_test.ndim == 1

        train, test = dask.compute(x_train, x_test)
        assert len(train) + len(test) == 1000
        assert set(train.flat) | set(test.flat) == set(range(1000))

    assert (first(KFold(3).split(X, y))[0].name ==
            first(KFold(3).split(X, y))[0].name)
    assert (first(KFold(4).split(X, y))[0].name !=
            first(KFold(3).split(X, y))[0].name)

    # Sparse
    X = X.map_partitions(sparse.csr_matrix, shape=X.shape, dtype=X.dtype)
    cv = list(KFold(4).split(X, None))
    x_train, y_train, x_test, y_test = cv[0]
    assert y_train is None and y_test is None
    train, test = dask.compute(x_train, x_test)
    assert train.shape[0] + test.shape[0] == 1000
    assert sparse.issparse(train) and sparse.issparse(test)

    with pytest.raises(ValueError):
        list(KFold(101).split(X, y))


def test_KFold():
    a = np.arange(1000)
    X = da.from_array(a[:, None], chunks=100)
    y = da.from_array(a, chunks=100)
    cv = list(KFold(4).split(X, y))
    assert len(cv) == 4
    cv2 = list(KFold(4).split(X, y))
    for p1, p2 in zip(cv, cv2):
        for i, j in zip(p1, p2):
            assert i.name == j.name
            assert i.dtype == j.dtype == a.dtype
        assert len(p1) == len(p2) == 4
        assert len(p1[0]) == len(p2[0]) == len(p1[1]) == len(p2[1]) == 750
        assert len(p1[2]) == len(p2[2]) == len(p1[3]) == len(p2[3]) == 250
        train_c, test_c = dask.compute(p1[0], p1[2])
        assert set(train_c.flat) | set(test_c.flat) == set(range(1000))
    assert cv[0][0].name != cv[1][0].name

    assert (first(KFold(4).split(X, y))[0].name !=
            first(KFold(3).split(X, y))[0].name)

    sets3 = list(KFold(4).split(X))
    X_train, y_train, X_test, y_test = sets3[0]
    assert len(X_train) + len(X_test) == 1000
    assert y_train is None and y_test is None

    with pytest.raises(ValueError):
        list(KFold(1001).split(X, y))

    with pytest.raises(ValueError):
        list(KFold(1).split(X, y))

    with pytest.raises(TypeError):
        list(KFold().split("not a dask collection"))


def test_DaskCVWrapper():
    X = np.arange(9)[:, None]
    y = np.arange(9)
    dcv = _DaskCVWrapper(LeavePOut(9, 3))
    sets = list(dcv.split(X, y))
    sets2 = list(dcv.split(X, y))
    assert len(sets) == len(dcv)
    for p1, p2 in zip(sets, sets2):
        for i, j in zip(p1, p2):
            assert i.key == j.key
    X_train, y_train, X_test, y_test = sets[0]
    train_c, test_c = dask.compute(X_train, X_test)
    assert set(train_c.flat) | set(test_c.flat) == set(range(9))
    dcv2 = _DaskCVWrapper(LeavePOut(9, 4))
    assert list(dcv2.split(X, y))[0][0].key != X_train.key

    sets = list(dcv2.split(X))
    assert sets[0][1] is None and sets[0][3] is None

    dX = da.from_array(X, chunks=3)
    with pytest.raises(TypeError):
        list(dcv2.split(dX))


def test_check_cv():
    X = np.arange(9)[:, None].repeat(5, 1)
    y = np.arange(9).repeat(5)
    dX = da.from_array(X, chunks=3)
    dy = da.from_array(y, chunks=3)

    cv = check_cv(None, dX, dy)
    assert isinstance(cv, KFold)
    assert cv.n_folds == 3

    cv = check_cv(2, dX, dy)
    assert isinstance(cv, KFold)
    assert cv.n_folds == 2

    cv = check_cv(None, X, y, True)
    assert isinstance(cv, _DaskCVWrapper)
    assert isinstance(cv.cv, StratifiedKFold)

    cv = check_cv(LeavePOut(3, 9), X, y)
    assert isinstance(cv, _DaskCVWrapper)
    assert isinstance(cv.cv, LeavePOut)

    rsplit = RandomSplit()
    cv = check_cv(rsplit, dX, dy)
    assert cv is rsplit

    with pytest.raises(ValueError):
        check_cv(rsplit, X, y)

    with pytest.raises(TypeError):
        check_cv("not a cv instance", dX, dy)


def test_train_test_split():
    x = np.arange(1000)
    a = da.from_array(x, chunks=100)
    m = dm.from_array(a)
    b = db.range(1000, npartitions=10)

    train_a, test_a = train_test_split(a, test_size=0.2, random_state=123)

    train_a2, test_a2, train, test = train_test_split(a, a + 10, test_size=0.2,
                                                      random_state=123)
    assert train_a2.name == train_a.name
    assert test_a2.name == test_a.name
    assert train_a2.name != train.name
    assert train_a.chunks == train.chunks
    assert test_a.chunks == test.chunks

    train_b, test_b, train_m, test_m = train_test_split(b, m, random_state=123)

    parts_b = train_b._get(train_b.dask, train_b._keys())
    parts_m = train_m._get(train_m.dask, train_m._keys())
    for p_b, p_m in zip(parts_b, parts_m):
        assert set(p_b) == set(p_m)

    with pytest.raises(ValueError):
        train_test_split(a, invalid_option=1)  # invalid kwargs

    with pytest.raises(ValueError):
        train_test_split(test_size=0.2)  # no arrays

    with pytest.raises(ValueError):
        train_test_split(a, b)  # not all da.Array

    with pytest.raises(ValueError):
        train_test_split(a, da.from_array(x, chunks=10))  # Not aligned

    with pytest.raises(ValueError):
        train_test_split(m, db.range(1000, npartitions=12))  # Not aligned
