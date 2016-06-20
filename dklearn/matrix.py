from __future__ import division, print_function, absolute_import

import numpy as np
from dask.base import Base, tokenize, normalize_token
from dask.compatibility import apply
from dask.threaded import get as threaded_get
from dask.utils import funcname
from scipy import sparse
from toolz import concat


def _vstack(mats):
    if len(mats) == 1:
        return mats[0]
    if sparse.issparse(mats[0]):
        return sparse.vstack(mats)
    return np.concatenate(mats)


class Matrix(Base):
    """A tall skinny matrix with optionally unknown shape.

    Can represent either sparse or dense matrices"""

    _optimize = staticmethod(lambda d, k, **kws: d)
    _default_get = staticmethod(threaded_get)

    def __init__(self, dask, name, npartitions, dtype=None, shape=None):
        self.dask = dask
        self.name = name
        self.npartitions = npartitions
        self.dtype = None if dtype is None else np.dtype(dtype)
        self.shape = shape

    def _keys(self):
        return [(self.name, i) for i in range(self.npartitions)]

    @staticmethod
    def _finalize(res):
        if len(res) == 1:
            return res[0]
        if sparse.issparse(res[0]):
            return sparse.vstack(res)
        return np.concatenate(res)

    @property
    def ndim(self):
        return len(self.shape) if self.shape is not None else None

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


def from_array(arr):
    name = 'matrix-from-array-' + tokenize(arr)
    if arr.ndim == 2:
        if len(arr.chunks[1]) != 1:
            arr = arr.rechunk((arr.chunks[0], arr.shape[1]))
        keys = list(concat(arr._keys()))
    elif arr.ndim == 1:
        keys = arr._keys()
    else:
        raise ValueError("array must be 1 or 2 dimensional")
    dsk = dict(((name, i), k) for i, k in enumerate(keys))
    dsk.update(arr.dask)
    return Matrix(dsk, name, len(keys), arr.dtype, arr.shape)


normalize_token.register(Matrix, lambda mat: mat.name)
