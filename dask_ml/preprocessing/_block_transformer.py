import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import compute
from pandas.api.types import is_categorical_dtype
from scipy import stats
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted, check_random_state

from dask_ml._utils import copy_learned_attributes
from dask_ml.utils import check_array, handle_zeros_in_scale


class BlockTransformer(FunctionTransformer):
    def __init__(
        self,
        func=None,
        inverse_func=None,
        validate=None,
        accept_sparse=False,
        check_inverse=True,
        kw_args=None,
        inv_kw_args=None,
    ):
        super(BlockTransformer, self).__init__(
            func=func,
            inverse_func=inverse_func,
            validate=validate,
            accept_sparse=accept_sparse,
            check_inverse=check_inverse,
            kw_args=kw_args,
            inv_kw_args=inv_kw_args,
        )

    def fit(self, X):
        return self

    def transform(self, X):
        if isinstance(X, da.Array):
            X = check_array(X, accept_dask_array=True, accept_unknown_chunks=True)
            XP = X.map_blocks(
                super(BlockTransformer, self).transform, dtype=X.dtype, chunks=X.chunks
            )
        else:
            X = check_array(X, accept_dask_array=False)
            XP = super(BlockTransformer, self).transform(X)
        return XP

    def inverse_transform(self, X):
        pass
