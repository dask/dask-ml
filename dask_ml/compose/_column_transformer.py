import dask.array as da
import dask.dataframe as dd
import numpy as np
import sklearn.compose
from scipy import sparse
from sklearn.compose._column_transformer import _get_transformer_list


class ColumnTransformer(sklearn.compose.ColumnTransformer):
    @staticmethod
    def _hstack(X, sparse_):
        """
        Stacks X horizontally.

        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames
        """
        if sparse_:
            return sparse.hstack(X).tocsr()
        elif any(isinstance(f, (dd.Series, dd.DataFrame)) for f in X):
            return dd.concat(X, axis="columns")
        elif any(isinstance(f, da.Array) for f in X):
            return da.hstack(X)
        else:
            return np.hstack(X)


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
