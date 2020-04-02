from typing import TypeVar

import numpy as np
from dask.array import Array
from pandas import Index, Series

# array-like

AnyArrayLike = TypeVar("AnyArrayLike", Index, Series, Array, np.ndarray)
ArrayLike = TypeVar("ArrayLike", Array, np.ndarray)
