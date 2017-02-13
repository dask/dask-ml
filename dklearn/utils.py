from __future__ import absolute_import, print_function, division

import dask.array as da
import dask.bag as db
from dask.base import Base, tokenize
from dask.delayed import Delayed
from dask.utils import concrete
from toolz import merge, concat
import sklearn.utils

from .core import DaskBaseEstimator


def is_dask_collection(x):
    return isinstance(x, (da.Array, db.Bag))


def unpack_arguments(*args):
    """Extracts dask values from args"""
    out_args = []
    dsks = []
    for x in args:
        t, d = unpack(x)
        out_args.append(t)
        dsks.extend(d)
    dsk = merge(dsks)
    return tuple(out_args) + (dsk,)


def unpack(expr):
    """Normalize a python object and extract all sub-dasks.

    Parameters
    ----------
    expr : object
        The object to be normalized.

    Returns
    -------
    task : normalized task to be run
    dasks : list of dasks that form the dag for this task
    """
    if isinstance(expr, Delayed):
        return expr.key, expr._dasks
    if isinstance(expr, Base):
        name = tokenize(expr, pure=True)
        keys = expr._keys()
        if isinstance(expr, DaskBaseEstimator):
            dsk = expr.dask
        else:
            dsk = expr._optimize(expr.dask, keys)
        dsk[name] = (expr._finalize, (concrete, keys))
        return name, [dsk]
    return expr, []


def check_X_y(X, y=False):
    has_y = y is not None and y is not False
    x_is_collection = is_dask_collection(X)
    y_is_collection = is_dask_collection(y)

    if has_y and (x_is_collection != y_is_collection):
        raise TypeError("X and y may not be mix of "
                        "non-dask and dask objects.""")
    if not x_is_collection and not y_is_collection:
        if has_y:
            sklearn.utils.check_consistent_length(X, y)
        return X, y

    x_is_array = isinstance(X, da.Array)
    y_is_array = isinstance(y, da.Array)

    if has_y and x_is_array != y_is_array:
        raise ValueError("If X is a da.Array, y must also be a da.Array")
    if x_is_array and y_is_array:
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        if y.ndim not in (1, 2):
            raise ValueError("y must be 1 or 2 dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must share first dimension")
        if X.chunks[0] != y.chunks[0]:
            raise ValueError("X and y chunks must be aligned")
    elif x_is_collection and y_is_collection:
        if X.npartitions != y.npartitions:
            raise ValueError("x and y must have the same number of partitions")

    if y is False:
        return X
    return X, y


def check_aligned_partitions(*arrays):
    if not arrays:
        return ()
    first = arrays[0]
    if isinstance(first, da.Array):
        if not all(isinstance(a, da.Array) for a in arrays):
            raise ValueError("Can't mix arrays and non-arrays")
        for a in arrays:
            if a.chunks[0] != first.chunks[0]:
                raise ValueError("All arguments must have chunks aligned")
    elif isinstance(first, db.Bag):
        for a in arrays:
            if a.npartitions != first.npartitions:
                raise ValueError("All arguments must have same npartitions")
    else:
        raise TypeError("Expected an instance of ``da.Array`` or ``db.Bag``,"
                        "got {0}".format(type(first).__name__))


def _unpack_keys_dask(x):
    if isinstance(x, da.Array):
        if x.ndim == 2:
            if len(x.chunks[1]) != 1:
                x = x.rechunk((x.chunks[0], x.shape[1]))
            keys = list(concat(x._keys()))
        else:
            keys = x._keys()
        dsk = x.dask
    elif isinstance(x, db.Bag):
        keys = x._keys()
        dsk = x.dask
    else:
        raise TypeError("Invalid input type: {0}".format(type(x)))
    return keys, dsk.copy()


def unpack_as_lists_of_keys(*args):
    parts, dsks = zip(*map(_unpack_keys_dask, args))
    if len(set(map(len, parts))) != 1:
        raise ValueError("inputs must all have the same number "
                         "of partitions along the first dimension")
    return tuple(parts) + (merge(dsks),)
