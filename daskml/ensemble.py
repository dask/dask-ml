"""
The model
"""
import numpy as np
import dask.array as da

from typing import Union

Array = Union[np.array, da.array]


class GradientBoostingRegressor:

    def __init__(self, learning_rate=0.1, max_depth=3):
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        pass

    def predict(self, X, y):
        pass


def _gbrt(X: Array, y: Array, alpha: float, m: int):
    for t in range(m):
        pass
