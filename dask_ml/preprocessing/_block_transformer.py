import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

from dask_ml.utils import check_array, handle_zeros_in_scale


class BlockTransformer(BaseEstimator, TransformerMixin):
    """Construct a transformer from a an arbitrary callable

    The BlockTransformer forwards the blocks of the X arguments to a user-defined
    function or function-object an returns the result of this operation.
    This is useful for stateless operations, that can be performed on the cell or
    block level, such as taking the log of frequencies. In general the transformer
    is not suitable for e.g. standardization tasks as this requires information for
    a complete columns

    Parameters
    ----------
    func : callable
        The callable to use for the transformation.

    validate : bool, optional default=True
         Indicate that the input X array should be checked before calling
        ``func``.

    kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to func.
    """

    def __init__(self, func=None, validate=True, kw_args=None):
        self.func = func
        self.validate = validate
        self.accept_sparse = False
        self.kw_args = kw_args

    def fit(self, X):
        return self

    def transform(self, X):
        def func_forwarded(arr):
            return self.func(arr, **(self.kw_args if self.kw_args else {}))

        if isinstance(X, da.Array):
            if self.validate:
                X = check_array(X, accept_dask_array=True, accept_unknown_chunks=True)
            XP = X.map_blocks(func_forwarded, dtype=X.dtype, chunks=X.chunks)
        else:
            if self.validate:
                X = check_array(X, accept_dask_array=False)
            XP = func_forwarded(X)
        return XP
