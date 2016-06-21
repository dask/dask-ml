from __future__ import absolute_import, print_function, division

import dask.bag as db
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn import feature_extraction
from sklearn.base import BaseEstimator

from . import matrix as dm
from .core import from_sklearn
from .estimator import _transform, WrapperMixin
from .utils import unpack_arguments


class _HashingBase(WrapperMixin, BaseEstimator):
    """Base class for FeatureHasher and HashingVectorizer"""
    def __init__(self, *args, **kwargs):
        self._est = est = self._base_type(*args, **kwargs)
        self._name = name = 'estimator-' + tokenize(est)
        self.dask = {name: est}

    @classmethod
    def from_sklearn(cls, est):
        if not isinstance(est, cls._base_type):
            raise TypeError("Expected {0}, "
                            "got {1}".format(cls._base_type.__name__,
                                             type(est).__name__))
        return cls(**est.get_params())

    def fit(self, X, y, **params):
        # no-op
        return self

    def transform(self, raw_X, y=None):
        name = 'transform-' + tokenize(self, raw_X)
        sk_est = self._est
        if isinstance(raw_X, db.Bag):
            dsk = dict(((name, i), (_transform, sk_est, k))
                       for (i, k) in enumerate(raw_X._keys()))
            dsk.update(raw_X.dask)
            return dm.Matrix(dsk, name, raw_X.npartitions, dtype=self.dtype,
                             shape=(None, self.n_features))
        raw_X, dsk = unpack_arguments(raw_X)
        dsk[name] = (_transform, sk_est, raw_X)
        return Delayed(name, [dsk])

    def _fit_transform(self, X, y, **kwargs):
        return self, self.transform(X, y, **kwargs)


class FeatureHasher(_HashingBase):
    _base_type = feature_extraction.FeatureHasher


class HashingVectorizer(_HashingBase):
    _base_type = feature_extraction.text.HashingVectorizer


from_sklearn.dispatch.register(feature_extraction.FeatureHasher,
                               FeatureHasher.from_sklearn)


from_sklearn.dispatch.register(feature_extraction.text.HashingVectorizer,
                               HashingVectorizer.from_sklearn)
