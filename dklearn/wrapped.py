from __future__ import absolute_import, print_function, division

from operator import getitem

import dask.array as da
import dask.bag as db
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn.base import clone, BaseEstimator
from toolz import merge
from textwrap import wrap

from .core import DaskBaseEstimator, from_sklearn
from .utils import unpack_arguments, unpack_as_lists_of_keys


def _fit(est, X, y, kwargs):
    return clone(est).fit(X, y, **kwargs)


def _predict(est, X):
    return est.predict(X)


def _score(est, X, y, kwargs):
    return est.score(X, y, **kwargs)


def _transform(est, X):
    return est.transform(X)


def _fit_transform(est, X, y, kwargs):
    est = clone(est)
    if hasattr(est, 'fit_transform'):
        fit = est
        tr = est.fit_transform(X, y, **kwargs)
    else:
        fit = est.fit(X, y, **kwargs)
        tr = est.transform(X)
    return fit, tr


class ClassProxy(object):
    def __init__(self, wrap, cls):
        self.wrap = wrap
        self.cls = cls

    @property
    def __name__(self):
        return self.wrap.__name__

    def __call__(self, *args, **kwargs):
        return self.wrap(self.cls(*args, **kwargs))


class WrapperMixin(DaskBaseEstimator):
    """Mixin class for dask estimators that wrap sklearn estimators"""
    @classmethod
    def _finalize(cls, res):
        return res[0]

    @property
    def _estimator_type(self):
        return self._est._estimator_type

    def get_params(self, deep=True):
        return self._est.get_params(deep=deep)

    def set_params(self, **params):
        est = clone(self._est).set_params(**params)
        return type(self).from_sklearn(est)

    def __getattr__(self, attr):
        if hasattr(self._est, attr):
            return getattr(self._est, attr)
        else:
            raise AttributeError("Attribute {0} either missing, or not "
                                 "computed yet. Try calling `.compute()`, "
                                 "and check again.".format(attr))

    def __setattr__(self, k, v):
        if k in ('_name', 'dask', '_est'):
            object.__setattr__(self, k, v)
        else:
            raise AttributeError("Attribute setting not permitted. "
                                 "Use `set_params` to change parameters")

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(self._est._get_param_names())
        o.update(i for i in dir(self._est) if i.endswith('_'))
        return list(o)


class Wrapped(WrapperMixin, BaseEstimator):
    """A class for wrapping a scikit-learn estimator.

    All operations done on this estimator are pure (no mutation), and are done
    lazily (if applicable). Calling `compute` results in the wrapped estimator.
    """

    def __init__(self, est, dask=None, name=None):
        if not isinstance(est, BaseEstimator):
            raise TypeError("Expected instance of BaseEstimator, "
                            "got {0}".format(type(est).__name__))
        if dask is None and name is None:
            name = est.__class__.__name__ + '-' + tokenize(est)
            dask = {name: est}
        elif dask is None or name is None:
            raise ValueError("Must provide both dask and name")
        self.dask = dask
        self._name = name
        self._est = est

    @property
    def __class__(self):
        return ClassProxy(type(self), type(self._est))

    def __repr__(self):
        class_name = type(self).__name__
        est = ''.join(map(str.strip, repr(self._est).splitlines()))
        return '\n'.join(wrap('{0}({1})'.format(class_name, est),
                              subsequent_indent=" "*10))

    @classmethod
    def from_sklearn(cls, est):
        """Wrap a scikit-learn estimator"""
        if isinstance(est, cls):
            return est
        return cls(est)

    def fit(self, X, y, **kwargs):
        name = 'fit-%s-%s' % (type(self._est).__name__,
                              tokenize(self, X, y, kwargs))
        X, y, dsk = unpack_arguments(X, y)
        dsk.update(self.dask)
        dsk[name] = (_fit, self._name, X, y, kwargs)
        return Wrapped(self._est, dsk, name)

    def predict(self, X):
        name = 'predict-%s-%s' % (type(self._est).__name__,
                                  tokenize(self, X))
        if isinstance(X, (da.Array, db.Bag)):
            keys, dsk = unpack_as_lists_of_keys(X)
            dsk.update(((name, i), (_predict, self._name, k))
                       for (i, k) in enumerate(keys))
            dsk.update(self.dask)
            return da.Array(dsk, name, (X.chunks[0],), 'i8')
        X, dsk = unpack_arguments(X)
        dsk[name] = (_predict, self._name, X)
        return Delayed(name, [dsk, self.dask])

    def score(self, X, y, **kwargs):
        name = 'score-%s-%s' % (type(self._est).__name__,
                                tokenize(self, X, y, kwargs))
        X, y, dsk = unpack_arguments(X, y)
        dsk[name] = (_score, self._name, X, y, kwargs)
        return Delayed(name, [dsk, self.dask])

    def transform(self, X):
        name = 'transform-%s-%s' % (type(self._est).__name__,
                                    tokenize(self, X))
        X, dsk = unpack_arguments(X)
        dsk[name] = (_transform, self._name, X)
        return Delayed(name, [dsk, self.dask])

    def _fit_transform(self, X, y, **kwargs):
        clsname = type(self._est).__name__
        token = tokenize(self, X, y, kwargs)
        fit_tr_name = 'fit-transform-%s-%s' % (clsname, token)
        fit_name = 'fit-%s-%s' % (clsname, token)
        tr_name = 'tr-%s-%s' % (clsname, token)
        X, y, dsk = unpack_arguments(X, y)
        dsk[fit_tr_name] = (_fit_transform, self._name, X, y, kwargs)
        dsk1 = merge({fit_name: (getitem, fit_tr_name, 0)}, dsk, self.dask)
        dsk2 = merge({tr_name: (getitem, fit_tr_name, 1)}, dsk, self.dask)
        return Wrapped(self._est, dsk1, fit_name), Delayed(tr_name, [dsk2])


from_sklearn.dispatch.register(BaseEstimator, Wrapped.from_sklearn)
