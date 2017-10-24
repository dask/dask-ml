from collections import OrderedDict
import multiprocessing

import dask.array as da
import dask.dataframe as dd
from dask import persist, compute
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import data as skdata
from sklearn.utils.validation import check_random_state

from dask_ml.utils import handle_zeros_in_scale, slice_columns


class StandardScaler(skdata.StandardScaler):

    def fit(self, X, y=None):
        self._reset()
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
        values = persist(*to_persist.values())
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
        super(MinMaxScaler, self).__init__(feature_range, copy)
        self.columns = columns

        if not copy:
            raise NotImplementedError()

    def fit(self, X, y=None):
        self._reset()
        to_persist = OrderedDict()
        feature_range = self.feature_range

        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature "
                             "range must be smaller than maximum.")

        _X = slice_columns(X, self.columns)
        data_min = _X.min(0)
        data_max = _X.max(0)
        data_range = data_max - data_min
        scale = ((feature_range[1] - feature_range[0]) /
                 handle_zeros_in_scale(data_range))

        to_persist["data_min_"] = data_min
        to_persist["data_max_"] = data_max
        to_persist["data_range_"] = data_range
        to_persist["scale_"] = scale
        to_persist["min_"] = feature_range[0] - data_min * scale
        to_persist["n_samples_seen_"] = np.nan

        values = persist(*to_persist.values())
        for k, v in zip(to_persist, values):
            setattr(self, k, v)
        return self

    def partial_fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X, y=None, copy=None):
        _X = slice_columns(X, self.columns)
        _X *= self.scale_
        _X += self.min_

        if isinstance(_X, dd.DataFrame) and self.columns:
            for column in self.columns:
                X[column] = _X[column]
            return X
        else:
            return _X

    def inverse_transform(self, X, y=None, copy=None):
        if not hasattr(self, "scale_"):
            raise Exception("This %(name)s instance is not fitted yet. "
                            "Call 'fit' with appropriate arguments before "
                            "using this method.")
        _X = slice_columns(X, self.columns)
        _X -= self.min_
        _X /= self.scale_

        if isinstance(_X, dd.DataFrame) and self.columns:
            for column in self.columns:
                X[column] = _X[column]
            return X
        else:
            return _X


class QuantileTransformer(skdata.QuantileTransformer):
    """Transforms features using quantile information.

    This implementation differs from the scikit-learn implementation
    by using approximate quantiles. The scikit-learn docstring follows.
    """
    __doc__ = __doc__ + '\n'.join(
        skdata.QuantileTransformer.__doc__.split("\n")[1:])

    def _check_inputs(self, X, accept_sparse_negative=False):
        if isinstance(X, (pd.DataFrame, dd.DataFrame)):
            X = X.values
        if isinstance(X, np.ndarray):
            C = len(X) // min(multiprocessing.cpu_count(), 2)
            X = da.from_array(X, chunks=C)

        rng = check_random_state(self.random_state)
        # TODO: non-float dtypes?
        # TODO: sparse arrays?
        # TODO: mix of sparse, dense?
        sample = rng.uniform(size=(5, X.shape[1])).astype(X.dtype)
        super(QuantileTransformer, self)._check_inputs(
            sample, accept_sparse_negative=accept_sparse_negative)
        return X

    def _sparse_fit(self, X, random_state):
        raise NotImplementedError

    def _dense_fit(self, X, random_state):
        references = self.references_ * 100
        quantiles = [da.percentile(col, references) for col in X.T]
        self.quantiles_, = compute(da.vstack(quantiles).T)

    def _transform(self, X, inverse=False):
        X = X.copy()  # ...
        transformed = [self._transform_col(X[:, feature_idx],
                                           self.quantiles_[:, feature_idx],
                                           inverse)
                       for feature_idx in range(X.shape[1])]
        return da.vstack(transformed).T

    def _transform_col(self, X_col, quantiles, inverse):
        if self.output_distribution == 'normal':
            output_distribution = 'norm'
        else:
            output_distribution = self.output_distribution
        output_distribution = getattr(stats, output_distribution)

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            X_col = X_col.map_blocks(output_distribution.cdf)

        lower_bounds_idx = (X_col - skdata.BOUNDS_THRESHOLD <
                            lower_bound_x)
        upper_bounds_idx = (X_col + skdata.BOUNDS_THRESHOLD >
                            upper_bound_x)
        if not inverse:
            # See the note in scikit-learn. This trick is to avoid
            # repeated extreme values
            X_col = 0.5 * (
                X_col.map_blocks(np.interp, quantiles, self.references_) -
                (-X_col).map_blocks(np.interp, -quantiles[::-1],
                                    -self.references_[::-1])
            )
        else:
            X_col = X_col.map_blocks(np.interp, self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y

        if not inverse:
            X_col = X_col.map_blocks(output_distribution.ppf)
            clip_min = output_distribution.ppf(skdata.BOUNDS_THRESHOLD -
                                               np.spacing(1))
            clip_max = output_distribution.ppf(1 - (skdata.BOUNDS_THRESHOLD -
                                                    np.spacing(1)))
            X_col = da.clip(X_col, clip_min, clip_max)

        return X_col


class RobustScaler(skdata.RobustScaler):

    def _check_array(self, X, copy):
        return X

    def fit(self, X, y=None):
        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        quantiles = [da.percentile(col, [q_min, 50., q_max]) for col in X.T]
        quantiles = da.vstack(quantiles).compute()
        self.center_ = quantiles[:, 1]
        self.scale_ = quantiles[:, 2] - quantiles[:, 0]
        self.scale_ = skdata._handle_zeros_in_scale(self.scale_, copy=False)
        return self
