from __future__ import absolute_import, print_function, division

import numpy as np
from dask.base import tokenize
from dask.delayed import delayed
from sklearn.base import clone, BaseEstimator, is_classifier

from . import matrix as dm
from .core import DaskBaseEstimator
from .estimator import Estimator
from .utils import unpack_arguments, unpack_as_lists_of_keys, check_X_y


_unique_chunk = delayed(np.unique, pure=True)


@delayed(pure=True)
def _unique_merge(x):
    return np.unique(np.concatenate(x))


def _partial_fit(est, X, y, classes, kwargs):
    # XXX: this mutates est!
    if classes is None:
        return est.partial_fit(X, y, **kwargs)
    return est.partial_fit(X, y, classes=classes, **kwargs)


class Chained(DaskBaseEstimator, BaseEstimator):
    _finalize = staticmethod(lambda res: Chained(res[0]))

    def __init__(self, estimator):
        if not isinstance(estimator, (BaseEstimator, Estimator)):
            raise TypeError("`estimator` must a scikit-learn estimator "
                            "or a dklearn.Estimator")

        if not hasattr(estimator, 'partial_fit'):
            raise ValueError("estimator must support `partial_fit`")

        est = Estimator.from_sklearn(estimator)
        object.__setattr__(self, 'estimator', est)

    @property
    def dask(self):
        return self.estimator.dask

    @property
    def _name(self):
        return self.estimator._name

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def set_params(self, **params):
        if not params:
            return self
        if 'estimator' in params:
            if len(params) == 1:
                return Chained(params['estimator'])
            raise ValueError("Setting params with both `'estimator'` and "
                             "nested parameters is ambiguous due to order of "
                             "operations. To change both estimator and "
                             "sub-parameters create a new `Chained`.")
        sub_params = {}
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1 and split[0] == 'estimator':
                sub_params[split[1]] = value
            else:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self.__class__.__name__))
        return Chained(self.estimator.set_params(**sub_params))

    def __setattr__(self, k, v):
        raise AttributeError("Attribute setting not permitted. "
                             "Use `set_params` to change parameters")

    @classmethod
    def from_sklearn(cls, est):
        if isinstance(est, cls):
            return est
        return cls(est)

    def to_sklearn(self, compute=True):
        return self.estimator.to_sklearn(compute=compute)

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y)
        name = 'partial_fit-' + tokenize(self, X, y, **kwargs)
        x_parts, y_parts, dsk = unpack_as_lists_of_keys(X, y)

        # Extract classes if applicable
        if is_classifier(self):
            classes = kwargs.pop('classes', None)
            if classes is None:
                classes = _unique_merge([_unique_chunk(i) for i in y_parts])
            classes, dsk2 = unpack_arguments(classes)
            dsk.update(dsk2)
        else:
            classes = None

        # Clone so that this estimator isn't mutated
        sk_est = clone(self.estimator._est)

        dsk[(name, 0)] = (_partial_fit, sk_est, x_parts[0], y_parts[0],
                          classes, kwargs)

        for i, (x, y) in enumerate(zip(x_parts[1:], y_parts[1:]), 1):
            dsk[(name, i)] = (_partial_fit, (name, i - 1), x, y, None, kwargs)
        out = Estimator(clone(sk_est), dsk, (name, len(x_parts) - 1))
        return Chained(out)

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y, **kwargs):
        return self.estimator.score(X, y, **kwargs)

    def transform(self, X):
        return self.estimator.transform(X)
