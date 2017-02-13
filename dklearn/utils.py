from __future__ import absolute_import, print_function, division

from dask.base import Base, tokenize
from dask.delayed import Delayed
from dask.utils import concrete
from toolz import merge

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
