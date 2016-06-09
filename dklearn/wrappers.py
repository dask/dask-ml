from __future__ import absolute_import, print_function, division

from sklearn.base import clone, BaseEstimator
from dask.base import Base, tokenize, normalize_token
from dask.optimize import fuse
from dask.threaded import get as threaded_get
from dask.delayed import Delayed
from toolz import first, partial


def _fit(est, X, y, kwargs):
    return clone(est).fit(X, y, **kwargs)


def _predict(est, X):
    return est.predict(X)


def _score(est, X, y, kwargs):
    return est.score(X, y, **kwargs)


def optimize(dsk, keys, **kwargs):
    dsk2, deps = fuse(dsk, keys)
    return dsk2


def from_sklearn(est):
    return Estimator.from_sklearn(est)


class Estimator(Base):
    """A class for wrapping a scikit-learn estimator.

    All operations done on this estimator are pure (no mutation), and are done
    lazily (if applicable). Calling `compute` results in the wrapped estimator.
    """
    _optimize = staticmethod(optimize)
    _default_get = staticmethod(threaded_get)
    _finalize = staticmethod(first)

    def __init__(self, dask, name, est):
        self.dask = dask
        self._name = name
        self._est = est

    def _keys(self):
        return [self._name]

    @classmethod
    def from_sklearn(cls, est):
        """Wrap a scikit-learn estimator"""
        name = 'from_sklearn-' + tokenize(est)
        return cls({name: est}, name, est)

    def get_params(self, deep=True):
        return self._est.get_params(deep=deep)

    def set_params(self, **params):
        est = clone(self._est).set_params(**params)
        return Estimator.from_sklearn(est)

    def fit(self, X, y, **kwargs):
        dsk = self.dask.copy()
        name = 'fit-' + tokenize(self, X, y, kwargs)
        dsk[name] = (_fit, self._name, X, y, kwargs)
        return Estimator(dsk, name, self._est)

    def predict(self, X):
        dsk = self.dask.copy()
        name = 'predict-' + tokenize(self, X)
        dsk[name] = (_predict, self._name, X)
        return Delayed(name, [dsk])

    def score(self, X, y, **kwargs):
        dsk = self.dask.copy()
        name = 'score-' + tokenize(self, X, y, kwargs)
        dsk[name] = (_score, self._name, X, y, kwargs)
        return Delayed(name, [dsk])


@partial(normalize_token.register, BaseEstimator)
def normalize_estimator(est):
    return type(est).__name__, normalize_token(vars(est))


@partial(normalize_token.register, Estimator)
def normalize_lazy_dask_estimator(est):
    return type(est).__name__, est._name
