from __future__ import absolute_import, print_function, division

import dask.array as da
import dask.bag as db
from dask.base import Base, tokenize
from dask.delayed import Delayed
from dask.utils import concrete
from toolz import merge, concat

from . import matrix as dm
from .core import DaskBaseEstimator


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
    x_is_array = isinstance(X, da.Array)
    y_is_array = isinstance(y, da.Array)

    if x_is_array and X.ndim != 2:
        raise ValueError("X must be 2 dimensional")
    if y_is_array and y.ndim not in (1, 2):
        raise ValueError("y must be 1 or 2 dimensional")
    if x_is_array and y_is_array:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must share first dimension")
        elif X.chunks[0] != y.chunks[0]:
            raise ValueError("X and y chunks must be aligned")
    if y is not None and y is not False:
        X_is_dask = isinstance(X, Base)
        if X_is_dask != isinstance(y, Base):
            raise TypeError("X and y may not be mix of "
                            "non-dask and dask objects.""")
        if X_is_dask and type(X) != type(y):
            raise TypeError("Dask type of X and y must match")
    if y is False:
        return X
    return X, y


def _unpack_keys_dask(x):
    if isinstance(x, da.Array):
        if x.ndim == 2:
            if len(x.chunks[1]) != 1:
                x = x.rechunk((x.chunks[0], x.shape[1]))
            keys = list(concat(x._keys()))
        else:
            keys = x._keys()
        dsk = x.dask
    elif isinstance(x, (db.Bag, dm.Matrix)):
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
