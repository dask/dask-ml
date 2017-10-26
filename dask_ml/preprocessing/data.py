from collections import OrderedDict
from distutils.version import LooseVersion
import multiprocessing

import dask.array as da
import dask.dataframe as dd
from dask import persist, compute
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import data as skdata
from sklearn.utils.validation import check_random_state

from dask_ml.utils import handle_zeros_in_scale, slice_columns

_PANDAS_VERSION = LooseVersion(pd.__version__)
_HAS_CTD = _PANDAS_VERSION >= '0.21.0'


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


class Categorizer(BaseEstimator, TransformerMixin):
    """Transform columns of a DataFrame to categoricals

    Parameters
    ----------
    categories : mapping, optional
        A dictionary mapping column name to instances of
        ``pandas.api.types.CategoricalDtype``. Alternatively, a
        mapping of column name to ``(categories, ordered)`` tuples.

    columns : sequence, optional
        A sequence of column names to limit the categorization to.
        This argument is ignored when ``categories`` is specified.

    Notes
    -----
    This transformer only applies to ``dask.DataFrame`` and
    ``pandas.DataFrame``. By default, all object-type columns are converted to
    categoricals. The set of categories will be the values present in the
    column and the categoricals will be unordered. Pass ``dtypes`` to control
    this behavior.

    Attributes
    ----------
    columns_ : pandas.Index
        The columns that were categorized. Useful when ``categories`` is None,
        and we detect the categorical and object columns

    categories_ : dict
        A dictionary mapping column names to dtypes. For pandas>=0.21.0, the
        values are instances of ``pandas.api.types.CategoricalDtype``. For
        older pandas, the values are tuples of ``(categories, ordered)``.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": ['a', 'a', 'b']})
    >>> ce = Categorizer()
    >>> ce.fit_transform(df).dtypes
    A       int64
    B    category
    dtype: object

    >>> ce.categories_
    {'B': CategoricalDtype(categories=['a', 'b'], ordered=False)}

    Using CategoricalDtypes for specifying the categories:

    >>> from pandas.api.types import CategoricalDtype
    >>> ce = Categorizer(categories={"B": CategoricalDtype(['a', 'b', 'c'])})
    >>> ce.fit_transform(df).B.dtype
    CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)
    """
    def __init__(self, categories=None, columns=None):
        self.categories = categories
        self.columns = columns

    def _check_array(self, X):
        # TODO: refactor to check_array
        if not isinstance(X, (pd.DataFrame, dd.DataFrame)):
            raise TypeError("Expected a pandas or dask DataFrame, got "
                            "{} instead".format(type(X)))
        return X

    def fit(self, X, y=None):
        """Find the categorical columns.

        """
        X = self._check_array(X)

        if self.categories is not None:
            # some basic validation
            columns = pd.Index(self.categories)
            categories = self.categories

        elif isinstance(X, pd.DataFrame):
            columns, categories = self._fit(X)
        else:
            columns, categories = self._fit_dask(X)

        self.columns_ = columns
        self.categories_ = categories
        return self

    def _fit(self, X):
        if self.columns is None:
            columns = X.select_dtypes(include=['object', 'category']).columns
        else:
            columns = self.columns
        categories = {}
        for name in columns:
            col = X[name]
            if not is_categorical_dtype(col):
                # This shouldn't ever be hit on a dask.array, since
                # the object columns would have been converted to known cats
                # already
                col = pd.Series(col, index=X.index).astype('category')

            if _HAS_CTD:
                categories[name] = col.dtype
            else:
                categories[name] = (col.cat.categories, col.cat.ordered)

        return columns, categories

    def _fit_dask(self, X):
        columns = self.columns
        df = X.categorize(columns=columns, index=False)
        return self._fit(df)

    def transform(self, X, y=None):
        """Transform the columns in ``X`` according to ``self.categories_``.
        """
        X = self._check_array(X).copy()
        categories = self.categories_

        for k, dtype in categories.items():
            if _HAS_CTD:
                if not isinstance(dtype, pd.api.types.CategoricalDtype):
                    dtype = pd.api.types.CategoricalDtype(*dtype)
                X[k] = X[k].astype(dtype)
            else:
                cat, ordered = dtype
                X[k] = X[k].astype('category').cat.set_categories(cat, ordered)

        return X
