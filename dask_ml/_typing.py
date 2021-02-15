from typing import Any, Sequence, TypeVar, Union

import dask.dataframe as dd
import numpy as np
from dask.array import Array
from pandas import DataFrame, Index, Series

try:
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any  # type: ignore


AnyArrayLike = TypeVar("AnyArrayLike", Index, Series, Array, np.ndarray)
ArrayLike = TypeVar("ArrayLike", Array, np.ndarray)
FrameOrSeriesUnion = Union[DataFrame, Series, dd.Series, dd.DataFrame]
SeriesType = Union[dd.Series, Series]
DataFrameType = Union[DataFrame, dd.DataFrame]
Number = Union[int, float, np.float64, np.int64, np.int32]
Int = Union[int, np.int64, np.int32]
NDArrayOrScalar = Union[np.ndarray, Sequence[Number], Number]
