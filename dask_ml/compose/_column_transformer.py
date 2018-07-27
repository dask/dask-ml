import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import sklearn.compose
from scipy import sparse
from sklearn.compose._column_transformer import _get_transformer_list


class ColumnTransformer(sklearn.compose.ColumnTransformer):
    def __init__(
        self,
        transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=1,
        transformer_weights=None,
        preserve_dataframe=True,
    ):
        self.preserve_dataframe = preserve_dataframe
        super(ColumnTransformer, self).__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
        )

    def _hstack(self, Xs):
        """
        Stacks X horizontally.

        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames
        """
        types = set(type(X) for X in Xs)

        if self.sparse_output_:
            return sparse.hstack(Xs).tocsr()
        elif dd.Series in types or dd.DataFrame in types:
            return dd.concat(Xs, axis="columns")
        elif da.Array in types:
            return da.hstack(Xs)
        elif self.preserve_dataframe and (pd.Series in types or pd.DataFrame in types):
            return pd.concat(Xs, axis="columns")
        else:
            return np.hstack(Xs)


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


make_column_transformer.__doc__ = getattr(
    sklearn.compose.make_column_transformer, "__doc__"
)
