from collections import OrderedDict

from dask import persist
import dask.array as da
import dask.dataframe as dd
from sklearn.preprocessing import data as skdata


def _handle_zeros_in_scale(scale):
    return scale


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
            scale_ = var_.copy()
            scale_[scale_ == 0] = 1
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


class MinMaxScaler(skdata.MinMaxScaler):

    def __init__(self, feature_range=(0, 1), copy=True, columns=None):
        super().__init__(feature_range, copy)
        self._columns = columns

        if not copy:
            raise NotImplementedError()

    def fit(self, X, y=None):
        self._reset()
        self._cache = dict()
        to_persist = OrderedDict()
        feature_range = self.feature_range

        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature "
                             "range must be smaller")

        if self._columns and isinstance(X, dd.DataFrame):
            X = X[self._columns]

        data_min = X.min(0)
        data_max = X.max(0)
        data_range = data_max - data_min
        scale = ((feature_range[1] - feature_range[0]) /
                 _handle_zeros_in_scale(data_range))

        to_persist["data_min_"] = data_min
        to_persist["data_max_"] = data_max
        to_persist["data_range_"] = data_range
        to_persist["scale_"] = scale
        to_persist["min_"] = feature_range[0] - data_min * scale
        to_persist["n_samples_seen_"] = len(X)

        values = persist(*to_persist.values(), cache=self._cache)
        for k, v in zip(to_persist, values):
            setattr(self, k, v)
        return self

    def partial_fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X, y=None, copy=None):
        if self._columns and isinstance(X, dd.DataFrame):
            X = X[self._columns]
        X *= self.scale_
        X += self.min_
        return X

    def inverse_transform(self, X, y=None, copy=None):
        if not hasattr(self, "min_") or not hasattr(self, "scale_"):
            raise Exception("This %(name)s instance is not fitted yet. "
                            "Call 'fit' with appropriate arguments before "
                            "using this method.")
        X -= self.min_
        X /= self.scale_
        return X
