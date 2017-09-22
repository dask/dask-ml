from collections import OrderedDict

from dask import compute, persist
import dask.array as da
from sklearn.preprocessing import data as skdata


class StandardScaler(skdata.StandardScaler):

    def fit(self, X, y=None):
        self._reset()
        self._cache = {}
        to_persist = OrderedDict()

        if self.with_mean:
            mean_ = X.mean(0)
            to_persist['mean_'] = mean_
        if self.with_std:
            var_ = X.var(0)
            scale_ = var_[var_ == 0] = 1
            scale_ = da.sqrt(scale_)
            to_persist['scale_'] = scale_
            to_persist['var_'] = var_

        to_persist['n_samples_seen_'] = len(X)
        values = persist(*to_persist.values(), cache=self._cache)
        for k, v in zip(to_persist, values):
            setattr(self, k, v)
        return self

    def partial_fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X, y=None, copy=None):
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_
        return X
