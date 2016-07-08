from __future__ import division, print_function, absolute_import

import numpy as np
from dask.base import Base, tokenize, normalize_token
from dask.delayed import Delayed
from dask.compatibility import apply
from dask.threaded import get as threaded_get
from dask.utils import funcname
from scipy import sparse as sp_sparse
from toolz import concat, merge


def _vstack(mats):
    if len(mats) == 1:
        return mats[0]
    if sp_sparse.issparse(mats[0]):
        return sp_sparse.vstack(mats)
    return np.concatenate(mats)


def as_2d_array(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return np.atleast_2d(x).T
    return x


def as_2d_sparse(x):
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    return sp_sparse.csr_matrix(x)


class Matrix(Base):
    """A tall skinny matrix with optionally unknown shape.

    Can represent either sparse or dense matrices"""

    _optimize = staticmethod(lambda d, k, **kws: d)
    _default_get = staticmethod(threaded_get)
    _finalize = staticmethod(_vstack)

    def __init__(self, dask, name, npartitions, dtype=None, shape=None):
        self.dask = dask
        self.name = name
        self.npartitions = npartitions
        self.dtype = None if dtype is None else np.dtype(dtype)
        if shape is None:
            shape = (None, None)
        elif len(shape) != 2:
            raise ValueError("Matrices must be 2 dimensional")
        self.shape = shape

    def _keys(self):
        return [(self.name, i) for i in range(self.npartitions)]

    @property
    def ndim(self):
        return 2

    def map_partitions(self, func, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        shape = kwargs.pop('shape', None)

        token = tokenize(self, func, args, kwargs, dtype, shape)
        new = '{0}-{1}'.format(funcname(func), token)
        old = self.name
        args = list(args)

        dsk = dict(((new, i), (apply, func, [(old, i)] + args, kwargs))
                   for i in range(self.npartitions))
        dsk.update(self.dask)
        return Matrix(dsk, new, self.npartitions, dtype, shape)


def from_bag(bag, dtype=None, shape=None):
    name = 'matrix-from-bag-' + tokenize(bag)
    dsk = dict(((name, i), (_vstack, k)) for i, k in enumerate(bag._keys()))
    dsk.update(bag.dask)
    return Matrix(dsk, name, bag.npartitions, dtype, shape)


def from_delayed(values, dtype=None, shape=None):
    if isinstance(values, Delayed):
        values = [values]
    name = 'matrix-from-delayed-' + tokenize(*values)
    dsk = merge(v.dask for v in values)
    dsk.update(((name, i), v.key) for i, v in enumerate(values))
    return Matrix(dsk, name, len(values), dtype, shape)


def from_series(s, sparse=False):
    name = 'matrix-from-series-' + tokenize(s, sparse)
    f = as_2d_sparse if sparse else as_2d_array
    dsk = dict(((name, i), (f, k)) for i, k in enumerate(s._keys()))
    dsk.update(s.dask)
    return Matrix(dsk, name, s.npartitions, s.dtype, (None, 1))


def from_array(arr, sparse=False):
    name = 'matrix-from-array-' + tokenize(arr, sparse)
    if arr.ndim == 2:
        if len(arr.chunks[1]) != 1:
            arr = arr.rechunk((arr.chunks[0], arr.shape[1]))
        keys = list(concat(arr._keys()))
        shape = arr.shape
    elif arr.ndim == 1:
        keys = arr._keys()
        shape = (arr.shape[0], 1)
    else:
        raise ValueError("array must be 1 or 2 dimensional")
    f = as_2d_sparse if sparse else as_2d_array
    dsk = dict(((name, i), (f, k)) for i, k in enumerate(keys))
    dsk.update(arr.dask)
    return Matrix(dsk, name, len(keys), arr.dtype, shape)


normalize_token.register(Matrix, lambda mat: mat.name)
