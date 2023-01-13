from typing import Any, List, Optional, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import sklearn.preprocessing

from .._compat import SKLEARN_1_1_X
from .._typing import ArrayLike, DataFrameType, DTypeLike, SeriesType
from ..base import DaskMLBaseMixin
from ..utils import check_array
from .label import _encode, _encode_dask_array


class OneHotEncoder(DaskMLBaseMixin, sklearn.preprocessing.OneHotEncoder):
    """Encode categorical integer features as a one-hot numeric array.

    .. versionadded:: 0.8.0

    .. note::

       This requires scikit-learn 0.20.0 or newer.

    The input to this transformer should be an array-like of integers, strings,
    or categoricals, denoting the values taken on by categorical (discrete)
    features. The features are encoded using a one-hot (aka 'one-of-K' or
    'dummy') encoding scheme. This creates a binary column for each category
    and returns a sparse matrix or dense array.

    By default, the encoder derives the categories based on

    1. For arrays, the unique values in each feature
    2. For DataFrames, the CategoricalDtype information for each feature

    Alternatively, for arrays, you can also specify the `categories` manually.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

    drop : None, default=None
        The option to drop one of the categories per feature is not yet supported.

    sparse_output : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    dtype : number type, default=np.float64
        Desired dtype of output.

    handle_unknown : 'error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). The option to
        ignore unknown categories is not currently implemented.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``).

    dtypes_ : list of dtypes
        For DataFrame input, the CategoricalDtype information associated
        with each feature. For arrays, this is a list of Nones.

    Notes
    -----
    There are a few differences from scikit-learn.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from dask_ml.preprocessing import OneHotEncoder
    >>> import numpy as np
    >>> import dask.array as da
    >>> enc = OneHotEncoder()
    >>> X = da.from_array(np.array([['A'], ['B'], ['A'], ['C']]), chunks=2)
    >>> enc.fit(X)
    ... # doctest: +ELLIPSIS
    OneHotEncoder(categorical_features=None, categories=None,
           dtype=<... 'numpy.float64'>, handle_unknown='error',
           n_values=None, sparse_output=True)

    >>> enc.categories_
    [array(['A', 'B', 'C'], dtype='<U1')]

    >>> enc.transform(X)
    dask.array<concatenate, shape=(4, 3), dtype=float64, chunksize=(2, 3)>
    """

    _legacy_mode = False

    def __init__(
        self,
        n_values: Optional[int] = None,
        categorical_features: Optional[pd.Categorical] = None,
        categories: Union[str, ArrayLike] = "auto",
        drop: Optional[bool] = None,
        sparse_output: bool = True,
        dtype: DTypeLike = np.float64,
        handle_unknown: str = "error",
    ):
        if drop is not None:
            raise NotImplementedError("drop != None is not implemented yet.")
        super(OneHotEncoder, self).__init__(
            categories=categories,
            sparse_output=sparse_output,
            dtype=dtype,
            handle_unknown=handle_unknown,
        )

    @classmethod
    def _get_param_names(cls: Any) -> List[str]:
        return [
            "categories",
            "drop",
            "dtype",
            "sparse_output",
            "dtype",
            "handle_unknown",
        ]

    def get_params(self, deep: bool = True):
        return super().get_params(deep)

    def fit(
        self,
        X: Union[ArrayLike, DataFrameType],
        y: Optional[Union[ArrayLike, SeriesType]] = None,
    ) -> "OneHotEncoder":
        if self.handle_unknown == "ignore":
            raise NotImplementedError("handle_unkown='ignore' is not implemented yet.")
        if self.handle_unknown != "error":
            msg = "handle_unknown must be 'error'." "got {0}.".format(
                self.handle_unknown
            )
            raise ValueError(msg)

        if isinstance(X, (pd.Series, pd.DataFrame)) or dask.is_dask_collection(X):
            self._fit(X, handle_unknown=self.handle_unknown)
        else:
            super(OneHotEncoder, self).fit(X, y=y)

        return self

    def _fit(
        self,
        X: Union[ArrayLike, DataFrameType],
        handle_unknown: str = "error",
        force_all_finite: bool = True,
        return_counts=False,
    ):
        X = self._validate_data(
            X, accept_dask_dataframe=True, dtype=None, preserve_pandas_dataframe=True
        )
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)

        if isinstance(X, np.ndarray):
            kwargs = {
                "handle_unknown": handle_unknown,
                "force_all_finite": force_all_finite,
            }

            # `return_counts` expected as of scikit-learn 1.1
            if SKLEARN_1_1_X:
                kwargs["return_counts"] = return_counts

            return super(OneHotEncoder, self)._fit(
                X,
                **kwargs,
            )

        is_array = isinstance(X, da.Array)

        if is_array:
            _, n_features = X.shape
        else:
            n_features = len(X.columns)

        if self.categories != "auto":
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet supported")
            if len(self.categories) != n_features:
                raise ValueError(
                    "Shape mismatch: if n_values is an array,"
                    " it has to be of shape (n_features,)."
                )

        self.categories_ = []
        self.dtypes_: List[Optional[pd.CategoricalDtype]] = []

        if is_array:
            for i in range(n_features):
                Xi = X[:, i]
                if self.categories == "auto":
                    cats = _encode(Xi)
                else:
                    cats = np.array(self.categories[i], dtype=X.dtype)
                self.categories_.append(cats)
                self.dtypes_.append(None)
        else:
            for i in range(len(X.columns)):
                Xi = X.iloc[:, i]
                if self.categories != "auto":
                    categories = self.categories[i]
                    Xi = Xi.astype(pd.CategoricalDtype(categories))
                else:
                    if not pd.api.types.is_categorical_dtype(Xi.dtype):
                        raise ValueError(
                            "All columns must be Categorical dtype when "
                            "'categories=\"auto\"'."
                        )

                cats = _encode(Xi, uniques=Xi.cat.categories)
                self.categories_.append(cats)
                self.dtypes_.append(Xi.dtype)

        self.categories_ = dask.compute(self.categories_)[0]

    def transform(
        self, X: Union[ArrayLike, DataFrameType]
    ) -> Union[ArrayLike, DataFrameType]:
        return self._transform(X)

    def _transform_new(
        self, X: Union[ArrayLike, DataFrameType]
    ) -> Union[ArrayLike, DataFrameType]:
        return self._transform(X)

    def _transform(
        self, X: Union[ArrayLike, DataFrameType], handle_unknown: str = "error"
    ) -> Union[ArrayLike, DataFrameType]:
        X = check_array(
            X, accept_dask_dataframe=True, dtype=None, preserve_pandas_dataframe=True
        )

        is_array = isinstance(X, da.Array)

        if is_array:
            _, n_features = X.shape
        else:
            n_features = len(X.columns)

        if is_array:
            # We encode each column independently, as they have different categories.
            Xs = [
                _encode_dask_array(
                    X[:, i],
                    uniques=self.categories_[i],
                    encode=True,
                    onehot_dtype=self.dtype,
                )[1]
                for i in range(n_features)
            ]
            X = da.concatenate(Xs, axis=1)

            if not self.sparse_output:
                X = X.map_blocks(lambda x: x.toarray(), dtype=self.dtype)

        else:
            import dask.dataframe as dd

            X = X.copy()

            if not len(X.columns) == len(self.categories_):
                raise ValueError(
                    "Number of columns ({}) does not match number "
                    "of categories_ ({})".format(len(X.columns), len(self.categories_))
                )

            for i, (col, dtype) in enumerate(zip(X.columns, self.dtypes_)):
                Xi = X.iloc[:, i]
                if not pd.api.types.is_categorical_dtype(Xi.dtype):
                    Xi = Xi.astype(dtype)
                    X[col] = Xi

                if Xi.dtype != dtype:
                    raise ValueError(
                        "Different CategoricalDtype for fit and transform. "
                        "{!r} != {!r}".format(Xi.dtype, dtype)
                    )
            return dd.get_dummies(X, sparse=self.sparse_output, dtype=self.dtype)

        return X
