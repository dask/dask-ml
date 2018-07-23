from packaging import version

import dask
import dask.array as da
import numpy as np
import pandas as pd
import scipy.sparse

import sklearn.preprocessing

from ..utils import check_array
from .._compat import SK_VERSION
from .label import _encode

# scikit-learn pre 0.20 had OneHotEncoder but we don't support its semantics.

if SK_VERSION < version.Version("0.20.0dev0"):
    raise ImportError("scikit-learn>= 0.20 required from OneHotEncoder")


class OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
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
        X = check_array(X, accept_dask_dataframe=True, dtype=None)
        # TODO
        # X_temp = check_array(X, accept_dask_dataframe=True, dtype=None)
        # if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
        #     X = check_array(X, dtype=np.object)
        # else:
        #     X = X_temp
        is_array = isinstance(X, da.Array)

        if is_array:
            _, n_features = X.shape
        else:
            if X.ndim == 1:
                X = X.to_frame()
            n_features = len(X.columns)

        if self.categories != "auto":
            # TODO
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
                    cats = np.array(self._categories[i], dtype=X.dtype)
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

        self.categories_ = dask.compute(self.categories_)[0]

    def _transform_new(self, X):
        # TODO: check_array
        is_array = isinstance(X, da.Array)

        if is_array:
            _, n_features = X.shape
        else:
            if X.ndim == 1:
                X = X.to_frame()
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
                raise ValueError

            for col, dtype in zip(X.columns, self.dtypes_):
                if not (X[col].dtype == dtype):
                    raise ValueError

            return dd.get_dummies(X, sparse=self.sparse, dtype=self.dtype)

        return X


def construct(x, categories):
    data = np.ones(len(x))
    rows = np.arange(len(x))
    columns = x.ravel()
    return scipy.sparse.csr_matrix(
        (data, (rows, columns)), shape=(len(x), len(categories))
    )
