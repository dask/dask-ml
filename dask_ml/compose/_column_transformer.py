import dask.array as da
import dask.dataframe as dd
import numpy as np
import sklearn.compose
from scipy import sparse
from sklearn.compose._column_transformer import (
    _check_key_type,
    _fit_transform_one,
    _get_transformer_list,
    _transform_one,
)
from sklearn.utils.validation import check_is_fitted

import six


class ColumnTransformer(sklearn.compose.ColumnTransformer):
    def _validate_remainder(self, X):
        """
        Validates ``remainder`` and defines ``_remainder`` targeting
        the remaining columns.
        """
        # TODO(https://github.com/dask/dask/pull/3212)
        # remove once dd.DataFrame.shape is implemented
        # The only difference is the isinstance(X, dd.DataFrame) below.
        is_transformer = (
            hasattr(self.remainder, "fit") or hasattr(self.remainder, "fit_transform")
        ) and hasattr(self.remainder, "transform")
        if self.remainder not in ("drop", "passthrough") and not is_transformer:
            raise ValueError(
                "The remainder keyword needs to be one of 'drop', "
                "'passthrough', or estimator. '%s' was passed instead" % self.remainder
            )

        if isinstance(X, dd.DataFrame):
            n_columns = len(X.columns)
        else:
            n_columns = X.shape[1]
        cols = []
        for _, _, columns in self.transformers:
            cols.extend(_get_column_indices(X, columns))
        remaining_idx = sorted(list(set(range(n_columns)) - set(cols))) or None

        self._remainder = ("remainder", self.remainder, remaining_idx)

    def fit_transform(self, X, y=None):
        # TODO: remove this once we have a compatible _hstack.
        # This is only impelmented here to use our _hstack.
        self._validate_remainder(X)
        self._validate_transformers()

        result = self._fit_transform(X, y, _fit_transform_one)

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)

        self._update_fitted_transformers(transformers)
        self._validate_output(Xs)

        return _hstack(list(Xs))

    def transform(self, X):
        # TODO: remove this once we have a compatible _hstack.
        # This is only impelmented here to use our _hstack.
        check_is_fitted(self, "transformers_")

        Xs = self._fit_transform(X, None, _transform_one, fitted=True)
        self._validate_output(Xs)

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return _hstack(list(Xs))


def make_column_transformer(*transformers, **kwargs):
    # This is identical to scikit-learn's. We're just using our
    # ColumnTransformer instead.
    n_jobs = kwargs.pop("n_jobs", 1)
    remainder = kwargs.pop("remainder", "drop")
    if kwargs:
        raise TypeError(
            'Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0])
        )
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(transformer_list, n_jobs=n_jobs, remainder=remainder)


def _get_column_indices(X, key):
    """
    Get feature column indices for input data X and key.

    For accepted values of `key`, see the docstring of _get_column

    """
    # TODO(https://github.com/dask/dask/pull/3212)
    # remove once dd.DataFrame.shape is implemented
    # This if / else is the only difference
    if isinstance(X, dd.DataFrame):
        n_columns = len(X.columns)
    else:
        n_columns = X.shape[1]

    if callable(key):
        key = key(X)

    if _check_key_type(key, int):
        if isinstance(key, int):
            return [key]
        elif isinstance(key, slice):
            return list(range(n_columns)[key])
        else:
            return list(key)

    elif _check_key_type(key, six.string_types):
        try:
            all_columns = list(X.columns)
        except AttributeError:
            raise ValueError(
                "Specifying the columns using strings is only "
                "supported for pandas DataFrames"
            )
        if isinstance(key, six.string_types):
            columns = [key]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is not None:
                start = all_columns.index(start)
            if stop is not None:
                # pandas indexing with strings is endpoint included
                stop = all_columns.index(stop) + 1
            else:
                stop = n_columns + 1
            return list(range(n_columns)[slice(start, stop)])
        else:
            columns = list(key)

        return [all_columns.index(col) for col in columns]

    elif hasattr(key, "dtype") and np.issubdtype(key.dtype, np.bool_):
        # boolean mask
        return list(np.arange(n_columns)[key])
    else:
        raise ValueError(
            "No valid specification of the columns. Only a "
            "scalar, list or slice of all integers or all "
            "strings, or boolean mask is allowed"
        )


def _hstack(X):
    """
    Stacks X horizontally.

    Supports input types (X): list of
        numpy arrays, sparse arrays and DataFrames
    """
    if any(sparse.issparse(f) for f in X):
        return sparse.hstack(X).tocsr()
    elif any(isinstance(f, (dd.Series, dd.DataFrame)) for f in X):
        return dd.concat(X, axis="columns")
    elif any(isinstance(f, da.Array) for f in X):
        return da.hstack(X)
    else:
        return np.hstack(X)
