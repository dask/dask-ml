import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse
from packaging import version

import sklearn.preprocessing

from .._compat import SK_VERSION
from ..utils import check_array
from .label import _encode

# scikit-learn pre 0.20 had OneHotEncoder but we don't support its semantics.

if SK_VERSION < version.Version("0.20.0dev0"):
    raise ImportError("scikit-learn>= 0.20 required from OneHotEncoder")


class OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
    """Encode categorical integer features as a one-hot numeric array.

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

    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    dtype : number type, default=np.float
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
           n_values=None, sparse=True)

    >>> enc.categories_
    [array(['A', 'B', 'C'], dtype='<U1')]

    >>> enc.transform(X)
    dask.array<concatenate, shape=(4, 3), dtype=float64, chunksize=(2, 3)>
    """

    _legacy_mode = False

    def __init__(
        self,
        n_values=None,
        categorical_features=None,
        categories="auto",
        sparse=True,
        dtype=np.float64,
        handle_unknown="error",
    ):
        super().__init__(
            n_values, categorical_features, categories, sparse, dtype, handle_unknown
        )

    def fit(self, X, y=None):
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

    def _fit(self, X, handle_unknown="error"):
        X = check_array(
            X, accept_dask_dataframe=True, dtype=None, preserve_pandas_dataframe=True
        )
        # TODO
        # X_temp = check_array(X, accept_dask_dataframe=True, dtype=None)
        # if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
        #     X = check_array(X, dtype=np.object)
        # else:
        #     X = X_temp
        if isinstance(X, np.ndarray):
            return super(OneHotEncoder, self)._fit(X, handle_unknown=handle_unknown)

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
        self.dtypes_ = []  # TODO: document

        if is_array:
            for i in range(n_features):
                Xi = X[:, i]
                if self.categories == "auto":
                    cats = _encode(Xi)
                else:
                    cats = np.array(self.categories[i], dtype=X.dtype)
                    if self.handle_unknown == "error":
                        # TODO: check unknown
                        # diff = _encode_check_unknown(Xi, cats)
                        # if diff:
                        #     msg = (
                        #         "Found unknown categories {0} in column {1}"
                        #         " during fit".format(diff, i)
                        #     )
                        #     raise ValueError(msg)
                        pass
                self.categories_.append(cats)
                self.dtypes_.append(None)
        else:
            if not (X.dtypes == "category").all():
                raise ValueError("All columns must be Categorical dtype.")
            if self.categories == "auto":
                for col in X.columns:
                    Xi = X[col]
                    cats = _encode(Xi, uniques=Xi.cat.categories)
                    self.categories_.append(cats)
                    self.dtypes_.append(Xi.dtype)
            else:
                raise ValueError(
                    "Cannot specify 'categories' with DataFrame input. "
                    "Use a categorical dtype instead."
                )

        self.categories_ = dask.compute(self.categories_)[0]

    def _transform_new(self, X):
        # TODO: check_array
        is_array = isinstance(X, da.Array)

        if is_array:
            _, n_features = X.shape
        else:
            n_features = len(X.columns)

        if is_array:
            # For arrays, the basic plan is:
            # User-array -> for each column:
            #     encoded dask array ->
            #     List[Delayed[sparse]] ->
            #     Array[sparse] -> (optionally Array[Dense]).
            Xs = [
                _encode(X[:, i], self.categories_[i], encode=True)[1]
                for i in range(n_features)
            ]

            objs = [
                [dask.delayed(construct)(block, categories) for block in x.to_delayed()]
                for x, categories in zip(Xs, self.categories_)
            ]
            arrs = []

            for i, (objs, x, categories) in enumerate(zip(objs, Xs, self.categories_)):
                inner_ars = [
                    # TODO: dtype
                    da.from_delayed(obj, (n_rows, len(categories)), dtype=self.dtype)
                    for j, (obj, n_rows) in enumerate(zip(objs, x.chunks[0]))
                ]
                arrs.append(da.concatenate(inner_ars))
                # TODO: this check fails. See why.
                # da.utils._check_dsk(arrs[-1].dask)

            X = da.concatenate(arrs, axis=1)

            if not self.sparse:
                X = X.map_blocks(lambda x: x.toarray(), dtype=self.dtype)

        else:
            import dask.dataframe as dd

            # Validate that all are categorical.
            if not (X.dtypes == "category").all():
                raise ValueError("Must be all categorical.")

            if not len(X.columns) == len(self.categories_):
                raise ValueError(
                    "Number of columns ({}) does not match number "
                    "of categories_ ({})".format(len(X.columns), len(self.categories_))
                )

            for col, dtype in zip(X.columns, self.dtypes_):
                if not (X[col].dtype == dtype):
                    raise ValueError(
                        "Different CategoricalDtype for fit and "
                        "transform. '{}' != {}'".format(dtype, X[col].dtype)
                    )

            return dd.get_dummies(X, sparse=self.sparse, dtype=self.dtype)

        return X


def construct(x, categories):
    data = np.ones(len(x))
    rows = np.arange(len(x))
    columns = x.ravel()
    return scipy.sparse.csr_matrix(
        (data, (rows, columns)), shape=(len(x), len(categories))
    )
