from typing import TypeVar, Union

import dask.dataframe as dd
import numpy as np
from dask.array import Array
from pandas import DataFrame, Index, Series

AnyArrayLike = TypeVar("AnyArrayLike", Index, Series, Array, np.ndarray)
ArrayLike = TypeVar("ArrayLike", Array, np.ndarray)
FrameOrSeriesUnion = Union[DataFrame, Series, dd.Series, dd.DataFrame]
SeriesType = Union[dd.Series, Series]
DataFrameType = Union[DataFrame, dd.DataFrame]
Number = Union[int, float, np.float, np.int]
Int = Union[int, np.int64, np.int32]
