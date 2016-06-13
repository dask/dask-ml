from __future__ import absolute_import, print_function, division

from functools import partial

from dask.base import Base, normalize_token, tokenize
from dask.delayed import Delayed
from dask.optimize import fuse
from dask.threaded import get as threaded_get
from dask.utils import concrete, Dispatch
from sklearn.base import clone, BaseEstimator
from toolz import merge, identity


class DaskBaseEstimator(Base):
    """Base class for dask-backed estimators"""
    _default_get = staticmethod(threaded_get)

    @staticmethod
    def _optimize(dsk, keys, **kwargs):
        dsk2, deps = fuse(dsk, keys)
        return dsk2

    def get_params(self, deep=True):
        return self._est.get_params(deep=deep)

    def set_params(self, **params):
        est = clone(self._est).set_params(**params)
        return type(self).from_sklearn(est)


@partial(normalize_token.register, BaseEstimator)
def normalize_BaseEstimator(est):
    return type(est).__name__, normalize_token(vars(est))


@partial(normalize_token.register, DaskBaseEstimator)
def normalize_dask_estimators(est):
    return type(est).__name__, est._name


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


def from_sklearn(est):
    """Wrap a scikit-learn estimator in a dask object."""
    return from_sklearn.dispatch(est)


from_sklearn.dispatch = Dispatch()
from_sklearn.dispatch.register(DaskBaseEstimator, identity)
