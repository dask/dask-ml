from typing import TypeVar, Union

import dask as dd
import numpy as np
from dask.array import Array
from pandas import Index, Series, DataFrame

# array-like

AnyArrayLike = TypeVar("AnyArrayLike", Index, Series, Array, np.ndarray)
ArrayLike = TypeVar("ArrayLike", Array, np.ndarray)
FrameOrSeriesUnion = Union[DataFrame, Series, dd.Series, dd.DataFrame]
SeriesType = Union[dd.Series, Series]
DataFrameType = Union[DataFrame, dd.DataFrame]