import dask
import numpy as np
from scipy import sparse

import sklearn.preprocessing

from .._compat import hstack, ones_like
from ..utils import check_array
from .label import LabelEncoder


class OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
    def fit(self, X, y=None):
        if self.handle_unknown != 'error':
            msg = (
                "handle_unknown must be 'error'."
                "got {0}.".format(self.handle_unknown)
            )
            raise ValueError(msg)

        self._fit(X, handle_unknown=self.handle_unknown)
        return self

    def _fit(self, X, handle_unknown="error"):
        X_temp = check_array(X, accept_dask_dataframe=True)

        if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        _, n_features = X.shape

        if self._categories != "auto":
            for cats in self._categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet " "supported")
            if len(self._categories) != n_features:
                raise ValueError(
                    "Shape mismatch: if n_values is an array,"
                    " it has to be of shape (n_features,)."
                )

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self._categories == "auto":
                le.fit(Xi)
            else:
                if handle_unknown == "error":
                    valid_mask = np.in1d(Xi, self._categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)
                le.classes_ = np.array(self._categories[i])

        self.categories_ = [le.classes_ for le in self._label_encoders_]

    def _transform_new(self, X):
        """New implementation assuming categorical input"""
        X_temp = check_array(X, dtype=None)

        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        X_int, X_mask = self._transform(X, handle_unknown=self.handle_unknown)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        feature_indices = np.cumsum(n_values)

        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = X_mask.sum(axis=1).cumsum()
        indptr = np.insert(indptr, 0, 0)
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csr_matrix((data, indices, indptr),
                                shape=(n_samples, feature_indices[-1]),
                                dtype=self.dtype)
        if not self.sparse:
            return out.toarray()
        else:
            return out

    def _transform(self, X, handle_unknown="error"):
        X_temp = check_array(X, accept_dask_dataframe=True, dtype=None)
        if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        _, n_features = X.shape

        if dask.is_dask_collection(X):
            X_int = hstack([
                self._label_encoders_[i].transform(X[:, i])
                for i in range(n_features)
            ])
            import pdb; pdb.set_trace()
            return X_int, ones_like(X, dtype=np.bool)
        else:
            return super(OneHotEncoder, self)._transform(X, handle_unknown=handle_unknown)

        return X_int, X_mask


__all__ = ["OneHotEncoder"]
