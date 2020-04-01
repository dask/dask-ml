from pathlib import Path
from typing import TypeVar

import numpy as np

from dask.array import Array

from pandas.core.arrays.base import ExtensionArray  # noqa: F401
from pandas.core.indexes.base import Index  # noqa: F401
from pandas.core.series import Series  # noqa: F401

# array-like

AnyArrayLike = TypeVar(
    "AnyArrayLike", "Index", "Series", "Array", np.ndarray
)
ArrayLike = TypeVar("ArrayLike", "Array", np.ndarray)
