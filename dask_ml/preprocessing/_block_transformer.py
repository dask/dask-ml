from typing import Any, Callable, Union

import dask.array as da
import dask.dataframe as dd
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dask_ml.utils import check_array

from .._typing import ArrayLike, DataFrameType, SeriesType


class BlockTransformer(BaseEstimator, TransformerMixin):
    """Construct a transformer from a an arbitrary callable

    The BlockTransformer forwards the blocks of the X arguments to a user-defined
    callable and returns the result of this operation.
    This is useful for stateless operations, that can be performed on the cell or
    block level, such as taking the log of frequencies. In general the transformer
    is not suitable for e.g. standardization tasks as this requires information for
    a complete column.

    Parameters
    ----------
    func : callable
        The callable to use for the transformation.

    validate : bool, optional default=False
         Indicate that the input X array should be checked before calling
        ``func``.

    kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to func.

    Examples
    --------

    >>> import dask.datasets
    >>> import pandas as pd
    >>> from dask_ml.preprocessing import BlockTransformer
    >>> df = dask.datasets.timeseries()
    >>> df
    ... # doctest: +SKIP
    Dask DataFrame Structure:
                       id    name        x        y
    npartitions=30
    2000-01-01      int64  object  float64  float64
    2000-01-02        ...     ...      ...      ...
    ...               ...     ...      ...      ...
    2000-01-30        ...     ...      ...      ...
    2000-01-31        ...     ...      ...      ...
    Dask Name: make-timeseries, 30 tasks
    >>> trn = BlockTransformer(pd.util.hash_pandas_object, index=False)
    >>> trn.transform(df)
    ... # doctest: +ELLIPSIS
    Dask Series Structure:
    npartitions=30
    2000-01-01    uint64
    2000-01-02       ...
                ...
    2000-01-30       ...
    2000-01-31       ...
    dtype: uint64
    Dask Name: hash_pandas_object, 60 tasks
    """

    def __init__(
        self,
        func: Callable[..., Union[ArrayLike, DataFrameType]],
        *,
        validate: bool = False,
        **kw_args: Any
    ):
        self.func = func
        self.validate = validate
        self.kw_args = kw_args

    def fit(
        self, X: Union[ArrayLike, DataFrameType], y: Union[ArrayLike, SeriesType] = None
    ) -> "BlockTransformer":
        return self

    def transform(
        self, X: Union[ArrayLike, DataFrameType], y: Union[ArrayLike, SeriesType] = None
    ) -> Union[ArrayLike, DataFrameType]:
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
