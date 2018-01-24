from __future__ import division

from collections import OrderedDict
from distutils.version import LooseVersion
import multiprocessing

import dask.array as da
import dask.dataframe as dd
from dask import compute
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import data as skdata
from sklearn.utils.validation import check_random_state, check_is_fitted

from dask_ml.utils import handle_zeros_in_scale, check_array

_PANDAS_VERSION = LooseVersion(pd.__version__)
_HAS_CTD = _PANDAS_VERSION >= '0.21.0'


class StandardScaler(skdata.StandardScaler):

    __doc__ = skdata.StandardScaler.__doc__

    def fit(self, X, y=None):
        self._reset()
        attributes = OrderedDict()

        if self.with_mean:
            mean_ = X.mean(0)
            attributes['mean_'] = mean_
        if self.with_std:
            var_ = X.var(0)
            scale_ = var_.copy()
            scale_[scale_ == 0] = 1
            scale_ = da.sqrt(scale_)
            attributes['scale_'] = scale_
            attributes['var_'] = var_

        attributes['n_samples_seen_'] = len(X)
        values = compute(*attributes.values())
        for k, v in zip(attributes, values):
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

    __doc__ = skdata.MinMaxScaler.__doc__

    def fit(self, X, y=None):
        self._reset()
        attributes = OrderedDict()
        feature_range = self.feature_range

        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature "
                             "range must be smaller than maximum.")

        data_min = X.min(0)
        data_max = X.max(0)
        data_range = data_max - data_min
        scale = ((feature_range[1] - feature_range[0]) /
                 handle_zeros_in_scale(data_range))

        attributes["data_min_"] = data_min
        attributes["data_max_"] = data_max
        attributes["data_range_"] = data_range
        attributes["scale_"] = scale
        attributes["min_"] = feature_range[0] - data_min * scale
        attributes["n_samples_seen_"] = np.nan

        values = compute(*attributes.values())
        for k, v in zip(attributes, values):
            setattr(self, k, v)
        return self

    def partial_fit(self, X, y=None):
        raise NotImplementedError()

    def transform(self, X, y=None, copy=None):
        # Workaround for https://github.com/dask/dask/issues/2840
        if isinstance(X, dd.DataFrame):
            X = X.mul(self.scale_).add(self.min_)
        else:
            X = X * self.scale_
            X = X + self.min_
        return X

    def inverse_transform(self, X, y=None, copy=None):
        if not hasattr(self, "scale_"):
            raise Exception("This %(name)s instance is not fitted yet. "
                            "Call 'fit' with appropriate arguments before "
                            "using this method.")
        X = X.copy()
        if isinstance(X, dd.DataFrame):
            X = X.sub(self.min_)
            X = X.div(self.scale_)
        else:
            X -= self.min_
            X /= self.scale_

        return X


class RobustScaler(skdata.RobustScaler):

    __doc__ = skdata.RobustScaler.__doc__

    def _check_array(self, X, *args, **kwargs):
        X = check_array(X, accept_dask_dataframe=True, **kwargs)
        return X

    def fit(self, X, y=None):
        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        if isinstance(X, dd.DataFrame):
            n_columns = len(X.columns)
            partition_lengths = X.map_partitions(len).compute()
            dtype = np.find_common_type(X.dtypes, [])
            blocks = X.to_delayed()
            X = da.vstack(
                [da.from_delayed(block.values, shape=(length, n_columns),
                                 dtype=dtype)
                 for block, length in zip(blocks, partition_lengths)])

        quantiles = [da.percentile(col, [q_min, 50., q_max]) for col in X.T]
        quantiles = da.vstack(quantiles).compute()
        self.center_ = quantiles[:, 1]
        self.scale_ = quantiles[:, 2] - quantiles[:, 0]
        self.scale_ = skdata._handle_zeros_in_scale(self.scale_, copy=False)
        return self


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
    """Transform columns of a DataFrame to categorical dtype.

    This is a useful pre-processing step for dummy, one-hot, or
    categorical encoding.

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

    All other columns are included in the transformed output untouched.

    For ``dask.DataFrame``, any unknown categoricals will become known.

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

        Parameters
        ----------
        X : pandas.DataFrame or dask.DataFrame
        y : ignored

        Returns
        -------
        self
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

        Parameters
        ----------
        X : pandas.DataFrame or dask.DataFrame
        y : ignored

        Returns
        -------
        X_trn : pandas.DataFrame or dask.DataFrame
            Same type as the input. The columns in ``self.categories_`` will
            be converted to categorical dtype.
        """
        check_is_fitted(self, "categories_")
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


class DummyEncoder(BaseEstimator, TransformerMixin):
    """Dummy (one-hot) encode categorical columns

    Parameters
    ----------
    columns : sequence, optional
        The columns to dummy encode. Must be categorical dtype.
        Dummy encodes all categorical dtype columns by default.
    drop_first : bool, default False
        Whether to drop the first category in each column.

    Attributes
    ----------
    columns_ : Index
        The columns in the training data before dummy encoding

    transformed_columns_ : Index
        The columns in the training data after dummy encoding

    categorical_columns_ : Index
        The categorical columns in the training data

    noncategorical_columns_ : Index
        The rest of the columns in the training data

    categorical_blocks_ : dict
        Mapping from column names to slice objects. The slices
        represent the positions in the transformed array that the
        categorical column ends up at

    dtypes_ : dict
        Dictionary mapping column name to either

        * instances of CategoricalDtype (pandas >= 0.21.0)
        * tuples of (categories, ordered)

    Notes
    -----
    This transformer only applies to dask and pandas DataFrames. For dask
    DataFrames, all of your categoricals should be known.

    The inverse transformation can be used on a dataframe or array.

    Examples
    --------
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4],
    ...                      "B": pd.Categorical(['a', 'a', 'a', 'b'])})
    >>> de = DummyEncoder()
    >>> trn = de.fit_transform(data)
    >>> trn
    A  B_a  B_b
    0  1    1    0
    1  2    1    0
    2  3    1    0
    3  4    0    1

    >>> de.columns_
    Index(['A', 'B'], dtype='object')

    >>> de.non_categorical_columns_
    Index(['A'], dtype='object')

    >>> de.categorical_columns_
    Index(['B'], dtype='object')

    >>> de.dtypes_
    {'B': CategoricalDtype(categories=['a', 'b'], ordered=False)}

    >>> de.categorical_blocks_
    {'B': slice(1, 3, None)}

    >>> de.fit_transform(dd.from_pandas(data, 2))
    Dask DataFrame Structure:
                    A    B_a    B_b
    npartitions=2
    0              int64  uint8  uint8
    2                ...    ...    ...
    3                ...    ...    ...
    Dask Name: get_dummies, 4 tasks
    """
    def __init__(self, columns=None, drop_first=False):
        self.columns = columns
        self.drop_first = drop_first

    def fit(self, X, y=None):
        """Determine the categorical columns to be dummy encoded.

        Parameters
        ----------
        X : pandas.DataFrame or dask.dataframe.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        columns = self.columns
        if columns is None:
            columns = X.select_dtypes(include=['category']).columns
        else:
            for column in columns:
                assert is_categorical_dtype(X[column]), "Must be categorical"

        self.categorical_columns_ = columns
        self.non_categorical_columns_ = X.columns.drop(
            self.categorical_columns_)

        if _HAS_CTD:
            self.dtypes_ = {col: X[col].dtype
                            for col in self.categorical_columns_}
        else:
            self.dtypes_ = {
                col: (X[col].cat.categories, X[col].cat.ordered)
                for col in self.categorical_columns_
            }

        left = len(self.non_categorical_columns_)
        self.categorical_blocks_ = {}
        for col in self.categorical_columns_:
            right = left + len(X[col].cat.categories)
            if self.drop_first:
                right -= 1
            self.categorical_blocks_[col], left = slice(left, right), right

        if isinstance(X, pd.DataFrame):
            sample = X.iloc[:1]
        else:
            sample = X._meta_nonempty

        self.transformed_columns_ = pd.get_dummies(
            sample, drop_first=self.drop_first).columns
        return self

    def transform(self, X, y=None):
        """Dummy encode the categorical columns in X

        Parameters
        ----------
        X : pd.DataFrame or dd.DataFrame
        y : ignored

        Returns
        -------
        transformed : pd.DataFrame or dd.DataFrame
            Same type as the input
        """
        if not X.columns.equals(self.columns_):
            raise ValueError("Columns of 'X' do not match the training "
                             "columns. Got {!r}, expected {!r}".format(
                                 X.columns, self.columns
                             ))
        if isinstance(X, pd.DataFrame):
            return pd.get_dummies(X, drop_first=self.drop_first)
        elif isinstance(X, dd.DataFrame):
            return dd.get_dummies(X, drop_first=self.drop_first)
        else:
            raise TypeError("Unexpected type {}".format(type(X)))

    def inverse_transform(self, X):
        """Inverse dummy-encode the columns in `X`

        Parameters
        ----------
        X : array or dataframe
            Either the NumPy, dask, or pandas version

        Returns
        -------
        data : DataFrame
            Dask array or dataframe will return a Dask DataFrame.
            Numpy array or pandas dataframe will return a pandas DataFrame
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.transformed_columns_)

        elif isinstance(X, da.Array):
            # later on we concat(..., axis=1), which requires
            # known divisions. Suboptimal, but I think unavoidable.
            unknown = np.isnan(X.chunks[0]).any()
            if unknown:
                lengths = da.atop(len, 'i', X[:, 0],
                                  'i', dtype='i8').compute()
                X = X.copy()
                chunks = (tuple(lengths), X.chunks[1])
                X._chunks = chunks

            X = dd.from_dask_array(X, columns=self.transformed_columns_)

        big = isinstance(X, dd.DataFrame)

        if big:
            chunks = np.array(X.divisions)
            chunks[-1] = chunks[-1] + 1
            chunks = tuple(chunks[1:] - chunks[:-1])

        non_cat = X[list(self.non_categorical_columns_)]

        cats = []
        for col in self.categorical_columns_:
            slice_ = self.categorical_blocks_[col]
            if _HAS_CTD:
                dtype = self.dtypes_[col]
                categories, ordered = dtype.categories, dtype.ordered
            else:
                categories, ordered = self.dtypes_[col]

            # use .values to avoid warning from pandas
            inds = X[list(X.columns[slice_])].values
            codes = inds.argmax(1)

            if self.drop_first:
                codes += 1
                codes[(inds == 0).all(1)] = 0

            if big:
                # dask
                codes._chunks = (chunks,)
                # Need a Categorical.from_codes for dask
                series = (dd.from_dask_array(codes, columns=col)
                          .astype('category')
                          .cat.set_categories(np.arange(len(categories)),
                                              ordered=ordered)
                          .cat.rename_categories(categories))
                # Bug in pandas <= 0.20.3 lost name
                if series.name is None:
                    series.name = col
                series.divisions = X.divisions
            else:
                # pandas
                series = pd.Series(pd.Categorical.from_codes(
                    codes, categories, ordered=ordered
                ), name=col)

            cats.append(series)
        if big:
            df = dd.concat([non_cat] + cats, axis=1)[list(self.columns_)]
        else:
            df = pd.concat([non_cat] + cats, axis=1)[self.columns_]
        return df


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal (integer) encode categorical columns.

    Parameters
    ----------
    columns : sequence, optional
        The columns to encode. Must be categorical dtype.
        Encodes all categorical dtype columns by default.

    Attributes
    ----------
    columns_ : Index
        The columns in the training data before/after encoding

    categorical_columns_ : Index
        The categorical columns in the training data

    noncategorical_columns_ : Index
        The rest of the columns in the training data

    dtypes_ : dict
        Dictionary mapping column name to either

        * instances of CategoricalDtype (pandas >= 0.21.0)
        * tuples of (categories, ordered)

    Notes
    -----
    This transformer only applies to dask and pandas DataFrames. For dask
    DataFrames, all of your categoricals should be known.

    The inverse transformation can be used on a dataframe or array.

    Examples
    --------
    >>> data = pd.DataFrame({"A": [1, 2, 3, 4],
    ...                      "B": pd.Categorical(['a', 'a', 'a', 'b'])})
    >>> enc = OrdinalEncoder()
    >>> trn = enc.fit_transform(data)
    >>> trn
       A  B
    0  1  0
    1  2  0
    2  3  0
    3  4  1

    >>> enc.columns_
    Index(['A', 'B'], dtype='object')

    >>> enc.non_categorical_columns_
    Index(['A'], dtype='object')

    >>> enc.categorical_columns_
    Index(['B'], dtype='object')

    >>> enc.dtypes_
    {'B': CategoricalDtype(categories=['a', 'b'], ordered=False)}

    >>> enc.fit_transform(dd.from_pandas(data, 2))
    Dask DataFrame Structure:
                       A     B
    npartitions=2
    0              int64  int8
    2                ...   ...
    3                ...   ...
    Dask Name: assign, 8 tasks

    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        """Determine the categorical columns to be encoded.

        Parameters
        ----------
        X : pandas.DataFrame or dask.dataframe.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        self.columns_ = X.columns
        columns = self.columns
        if columns is None:
            columns = X.select_dtypes(include=['category']).columns
        else:
            for column in columns:
                assert is_categorical_dtype(X[column]), "Must be categorical"

        self.categorical_columns_ = columns
        self.non_categorical_columns_ = X.columns.drop(
            self.categorical_columns_)

        if _HAS_CTD:
            self.dtypes_ = {col: X[col].dtype
                            for col in self.categorical_columns_}
        else:
            self.dtypes_ = {
                col: (X[col].cat.categories, X[col].cat.ordered)
                for col in self.categorical_columns_
            }

        return self

    def transform(self, X, y=None):
        """Ordinal encode the categorical columns in X

        Parameters
        ----------
        X : pd.DataFrame or dd.DataFrame
        y : ignored

        Returns
        -------
        transformed : pd.DataFrame or dd.DataFrame
            Same type as the input
        """
        if not X.columns.equals(self.columns_):
            raise ValueError("Columns of 'X' do not match the training "
                             "columns. Got {!r}, expected {!r}".format(
                                 X.columns, self.columns
                             ))
        if not isinstance(X, (pd.DataFrame, dd.DataFrame)):
            raise TypeError("Unexpected type {}".format(type(X)))

        X = X.copy()
        for col in self.categorical_columns_:
            X[col] = X[col].cat.codes
        return X

    def inverse_transform(self, X):
        """Inverse ordinal-encode the columns in `X`

        Parameters
        ----------
        X : array or dataframe
            Either the NumPy, dask, or pandas version

        Returns
        -------
        data : DataFrame
            Dask array or dataframe will return a Dask DataFrame.
            Numpy array or pandas dataframe will return a pandas DataFrame
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)

        elif isinstance(X, da.Array):
            # later on we concat(..., axis=1), which requires
            # known divisions. Suboptimal, but I think unavoidable.
            unknown = np.isnan(X.chunks[0]).any()
            if unknown:
                lengths = da.atop(len, 'i', X[:, 0],
                                  'i', dtype='i8').compute()
                X = X.copy()
                chunks = (tuple(lengths), X.chunks[1])
                X._chunks = chunks

            X = dd.from_dask_array(X, columns=self.columns_)

        big = isinstance(X, dd.DataFrame)

        if big:
            chunks = np.array(X.divisions)
            chunks[-1] = chunks[-1] + 1
            chunks = tuple(chunks[1:] - chunks[:-1])

        X = X.copy()
        for col in self.categorical_columns_:
            if _HAS_CTD:
                dtype = self.dtypes_[col]
                categories, ordered = dtype.categories, dtype.ordered
            else:
                categories, ordered = self.dtypes_[col]

            # use .values to avoid warning from pandas
            codes = X[col].values

            if big:
                # dask
                codes._chunks = (chunks,)
                # Need a Categorical.from_codes for dask
                series = (dd.from_dask_array(codes, columns=col)
                          .astype('category')
                          .cat.set_categories(np.arange(len(categories)),
                                              ordered=ordered)
                          .cat.rename_categories(categories))
                # Bug in pandas <= 0.20.3 lost name
                if series.name is None:
                    series.name = col
                series.divisions = X.divisions
            else:
                # pandas
                series = pd.Series(pd.Categorical.from_codes(
                    codes, categories, ordered=ordered
                ), name=col)

            X[col] = series

        return X
