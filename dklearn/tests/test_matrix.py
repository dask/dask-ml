from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
import dask.array as da
import dask.bag as db
from dask.base import normalize_token
from dask.delayed import delayed
from scipy import sparse

import dklearn.matrix as dm


sp_mats = [sparse.csr_matrix([[1], [2], [3]]),
           sparse.csr_matrix([[4], [5], [6], [7]]),
           sparse.csr_matrix([[8], [9], [10]])]
sp_sol = sparse.vstack(sp_mats)

np_mats = [np.array([1, 2, 3]),
           np.array([4, 5, 6, 7]),
           np.array([8, 9, 10])]
np_sol = np.concatenate(np_mats)

np2d_mats = [np.expand_dims(a, 1) for a in np_mats]
np2d_sol = np.concatenate(np2d_mats)


def eq(a, b):
    a_sparse = sparse.issparse(a)
    b_sparse = sparse.issparse(b)
    if a_sparse != b_sparse:
        return False
    a = a.toarray() if a_sparse else a
    b = b.toarray() if b_sparse else b
    return (a == b).all()


@pytest.mark.parametrize(('mats', 'sol'), [(sp_mats, sp_sol),
                                           (np_mats, np_sol),
                                           (np2d_mats, np2d_sol)])
def test_matrix(mats, sol):
    dsk = dict((('test', i), m) for i, m in enumerate(mats))
    mat = dm.Matrix(dsk, 'test', 3)
    assert normalize_token(mat) == mat.name
    res = mat.compute()
    assert eq(res, sol)


def test_from_bag():
    b = db.from_sequence(sp_mats)
    mat = dm.from_bag(b)
    assert eq(mat.compute(), sp_sol)
    assert mat.ndim is None

    mat = dm.from_bag(b, dtype='i8', shape=sp_sol.shape)
    assert mat.dtype == np.dtype('i8')
    assert mat.shape == sp_sol.shape
    assert mat.ndim == 2
    assert eq(mat.compute(), sp_sol)


def test_from_delayed():
    darange = delayed(np.arange)
    x1 = darange(10)
    x2 = darange(10, 15)
    x3 = darange(15, 30)

    mat = dm.from_delayed([x1, x2, x3])

    assert mat.name == dm.from_delayed([x1, x2, x3]).name
    assert mat.name != dm.from_delayed([x2, x1, x3]).name

    assert mat.dtype is None
    assert mat.shape is None
    assert mat.ndim is None
    assert eq(mat.compute(), np.arange(30))

    mat = dm.from_delayed([x1, x2, x3], dtype='i8', shape=(None,))
    assert mat.dtype == np.dtype('i8')
    assert mat.shape == (None,)
    assert mat.ndim == 1
    assert eq(mat.compute(), np.arange(30))

    mat = dm.from_delayed(x1)
    assert eq(mat.compute(), x1.compute())


def test_from_series():
    dd = pytest.importorskip('dask.dataframe')
    import pandas as pd

    x = np.arange(30)
    s = pd.Series(x)
    ds = dd.from_pandas(s, npartitions=3)

    mat = dm.from_series(ds)
    assert mat.dtype == s.dtype
    assert mat.shape == (None,)
    assert eq(mat.compute(), x)


@pytest.mark.parametrize(('mats', 'sol'), [(sp_mats, sp_sol),
                                           (np_mats, np_sol),
                                           (np2d_mats, np2d_sol)])
def test_from_bag_multiple_in_partitions(mats, sol):
    b = db.Bag({('b', 0): mats[:2], ('b', 1): [mats[2]]}, 'b', 2)
    mat = dm.from_bag(b)
    assert eq(mat.compute(), sol)


@pytest.mark.parametrize('arr', [np_sol, np2d_sol])
def test_from_array(arr):
    a = da.from_array(arr, chunks=3)
    mat = dm.from_array(a)
    assert mat.name == dm.from_array(a).name
    assert mat.name != dm.from_array(a + 1).name
    assert mat.dtype == arr.dtype
    assert mat.shape == arr.shape
    assert mat.ndim == arr.ndim
    assert eq(mat.compute(), arr)


def test_from_array_wide():
    x = np.arange(100).reshape((10, 10))
    a = da.from_array(x, chunks=5)
    mat = dm.from_array(a)
    assert mat.name == dm.from_array(a).name
    assert mat.name != dm.from_array(a + 1).name
    assert mat.dtype == x.dtype
    assert mat.shape == x.shape
    assert mat.ndim == x.ndim
    assert eq(mat.compute(), x)


def test_from_array_3d_errors():
    x = np.arange(24).reshape((2, 3, 4))
    a = da.from_array(x, chunks=2)
    with pytest.raises(ValueError):
        dm.from_array(a)


def test_map_partitions():
    dsk = dict((('test', i), m) for i, m in enumerate(np_mats))
    mat = dm.Matrix(dsk, 'test', 3)

    def inc(x):
        return x + 1

    res = mat.map_partitions(inc)
    assert res.shape == mat.shape
    assert eq(res.compute(), np_sol + 1)

    def foo(mat, a, b=0):
        return mat + a + b

    res = mat.map_partitions(foo, 1, b=2)
    assert res.name == mat.map_partitions(foo, 1, b=2).name
    assert res.name != mat.map_partitions(foo, 2, b=2).name
    assert eq(res.compute(), np_sol + 1 + 2)

    res2 = mat.map_partitions(foo, 1, b=2, dtype='i8', shape=(10,))
    assert res2.name != res.name
    assert res2.ndim == 1
    assert res2.shape == (10,)
    assert res2.dtype == np.dtype('i8')
    assert eq(res2.compute(), np_sol + 1 + 2)
