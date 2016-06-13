from __future__ import absolute_import, print_function, division

from operator import getitem

from sklearn.base import clone, BaseEstimator
from sklearn import pipeline
from dask.base import Base, tokenize, normalize_token
from dask.optimize import fuse
from dask.threaded import get as threaded_get
from dask.delayed import Delayed
from dask.utils import concrete
from toolz import partial, merge


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


def optimize(dsk, keys, **kwargs):
    dsk2, deps = fuse(dsk, keys)
    return dsk2


def from_sklearn(est):
    if isinstance(est, pipeline.Pipeline):
        return Pipeline.from_sklearn(est)
    return Estimator.from_sklearn(est)


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


class DaskBaseEstimator(Base):
    """Base class for dask-backed estimators"""

    def get_params(self, deep=True):
        return self._est.get_params(deep=deep)

    def set_params(self, **params):
        est = clone(self._est).set_params(**params)
        return type(self).from_sklearn(est)


class Estimator(DaskBaseEstimator):
    """A class for wrapping a scikit-learn estimator.

    All operations done on this estimator are pure (no mutation), and are done
    lazily (if applicable). Calling `compute` results in the wrapped estimator.
    """
    _optimize = staticmethod(optimize)
    _default_get = staticmethod(threaded_get)

    def __init__(self, dask, name, est):
        self.dask = dask
        self._name = name
        self._est = est

    @staticmethod
    def _finalize(res):
        return res[0]

    def _keys(self):
        return [self._name]

    @classmethod
    def from_sklearn(cls, est):
        """Wrap a scikit-learn estimator"""
        name = 'from_sklearn-' + tokenize(est)
        return cls({name: est}, name, est)

    def fit(self, X, y, **kwargs):
        name = 'fit-' + tokenize(self, X, y, kwargs)
        X, y, dsk = unpack_arguments(X, y)
        dsk.update(self.dask)
        dsk[name] = (_fit, self._name, X, y, kwargs)
        return Estimator(dsk, name, self._est)

    def predict(self, X):
        name = 'predict-' + tokenize(self, X)
        X, dsk = unpack_arguments(X)
        dsk[name] = (_predict, self._name, X)
        return Delayed(name, [dsk, self.dask])

    def score(self, X, y, **kwargs):
        name = 'score-' + tokenize(self, X, y, kwargs)
        X, y, dsk = unpack_arguments(X, y)
        dsk[name] = (_score, self._name, X, y, kwargs)
        return Delayed(name, [dsk, self.dask])

    def transform(self, X):
        name = 'transform-' + tokenize(self, X)
        X, dsk = unpack_arguments(X)
        dsk[name] = (_transform, self._name, X)
        return Delayed(name, [dsk, self.dask])

    def _fit_transform(self, X, y, **kwargs):
        token = tokenize(self, X, y, kwargs)
        fit_tr_name = 'fit-transform-' + token
        fit_name = 'fit-' + token
        tr_name = 'tr-' + token
        X, y, dsk = unpack_arguments(X, y)
        dsk[fit_tr_name] = (_fit_transform, self._name, X, y, kwargs)
        dsk1 = merge({fit_name: (getitem, fit_tr_name, 0)}, dsk, self.dask)
        dsk2 = merge({tr_name: (getitem, fit_tr_name, 1)}, dsk, self.dask)
        return Estimator(dsk1, fit_name, self._est), Delayed(tr_name, [dsk2])


class Pipeline(DaskBaseEstimator):
    _optimize = staticmethod(optimize)
    _default_get = staticmethod(threaded_get)

    def __init__(self, steps, est):
        self.steps = steps
        self._name = 'pipeline-' + tokenize('Pipeline', steps)
        self._est = est

    @staticmethod
    def _finalize(res):
        return pipeline.Pipeline(res[0])

    @property
    def dask(self):
        if hasattr(self, '_dask'):
            return self._dask
        dsk = {}
        names = []
        tasks = []
        for n, s in self.steps:
            dsk.update(s.dask)
            names.append(n)
            tasks.append((s._finalize, s._keys()))
        dsk[self._name] = (list, (zip, names, tasks))
        self._dask = dsk
        return dsk

    def _keys(self):
        return [self._name]

    @classmethod
    def from_sklearn(cls, est):
        steps = [(k, from_sklearn(v)) for k, v in est.steps]
        return cls(steps, est)

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    @property
    def _estimator_type(self):
        return self._final_estimator._estimator_type

    def _pre_transform(self, Xt, y=None, **fit_params):
        # Separate out parameters
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        # Call fit_transform on all but last estimator
        fit_steps = []
        for name, transform in self.steps[:-1]:
            kwargs = fit_params_steps[name]
            fit, Xt = transform._fit_transform(Xt, y, **kwargs)
            fit_steps.append(fit)
        return fit_steps, Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        fit_steps, Xt, params = self._pre_transform(X, y, **fit_params)
        fit = self._final_estimator.fit(Xt, y, **params)
        fit_steps.append(fit)
        new_steps = [(old[0], s) for old, s in zip(self.steps, fit_steps)]
        return Pipeline(new_steps, self._est)

    def transform(self, X):
        for name, transform in self.steps:
            X = transform.transform(X)
        return X

    def predict(self, X):
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        return self._final_estimator.predict(X)

    def score(self, X, y=None):
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        return self._final_estimator.score(X, y)

    def _fit_transform(self, X, y=None, **fit_params):
        fit_steps, Xt, params = self._pre_transform(X, y, **fit_params)
        fit, Xt = self._final_estimator._fit_transform(X, y, **params)
        fit_steps.append(fit)
        new_steps = [(old[0], s) for old, s in zip(self.steps, fit_steps)]
        return Pipeline(new_steps, self._est), Xt


@partial(normalize_token.register, BaseEstimator)
def normalize_BaseEstimator(est):
    return type(est).__name__, normalize_token(vars(est))


@partial(normalize_token.register, DaskBaseEstimator)
def normalize_dask_estimators(est):
    return type(est).__name__, est._name
