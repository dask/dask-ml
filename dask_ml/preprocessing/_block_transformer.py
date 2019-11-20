import dask.array as da
import dask.dataframe as dd
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dask_ml.utils import check_array, handle_zeros_in_scale


class BlockTransformer(BaseEstimator, TransformerMixin):
    """Construct a transformer from a an arbitrary callable

    The BlockTransformer forwards the blocks of the X arguments to a user-defined
    function or function-object an returns the result of this operation.
    This is useful for stateless operations, that can be performed on the cell or
    block level, such as taking the log of frequencies. In general the transformer
    is not suitable for e.g. standardization tasks as this requires information for
    a complete column

    Parameters
    ----------
    func : callable
        The callable to use for the transformation.

    validate : bool, optional default=False
         Indicate that the input X array should be checked before calling
        ``func``.

    kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to func.
    """

    def __init__(self, func, *, validate=False, kw_args=None):
        self.func = func
        self.validate = validate
        self.kw_args = kw_args

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        kwargs = self.kw_args if self.kw_args else {}

        if isinstance(X, da.Array):
            if self.validate:
                X = check_array(X, accept_dask_array=True, accept_unknown_chunks=True)
            XP = X.map_blocks(self.func, dtype=X.dtype, chunks=X.chunks, **kwargs)
        elif isinstance(X, dd.DataFrame):
            if self.validate:
                X = check_array(
                    X, accept_dask_dataframe=True, preserve_pandas_dataframe=True
                )
            XP = X.map_partitions(self.func, **kwargs)
        elif isinstance(X, pd.DataFrame):
            if self.validate:
                X = check_array(
                    X, accept_dask_array=False, preserve_pandas_dataframe=True
                )
            XP = self.func(X, **kwargs)
        else:
            if self.validate:
                X = check_array(X, accept_dask_array=False)
            XP = self.func(X, **kwargs)
        return XP
