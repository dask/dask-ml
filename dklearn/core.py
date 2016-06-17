from __future__ import absolute_import, print_function, division

from functools import partial

from dask.base import Base, normalize_token
from dask.optimize import fuse
from dask.threaded import get as threaded_get
from dask.utils import Dispatch
from sklearn.base import BaseEstimator
from toolz import identity


class DaskBaseEstimator(Base):
    """Base class for dask-backed estimators"""
    _default_get = staticmethod(threaded_get)

    @staticmethod
    def _optimize(dsk, keys, **kwargs):
        dsk2, deps = fuse(dsk, keys)
        return dsk2

    def _keys(self):
        return [self._name]


@partial(normalize_token.register, BaseEstimator)
def normalize_BaseEstimator(est):
    return type(est).__name__, normalize_token(vars(est))


@partial(normalize_token.register, DaskBaseEstimator)
def normalize_dask_estimators(est):
    return type(est).__name__, est._name


def from_sklearn(est):
    """Wrap a scikit-learn estimator in a dask object."""
    return from_sklearn.dispatch(est)


from_sklearn.dispatch = Dispatch()
from_sklearn.dispatch.register(DaskBaseEstimator, identity)
