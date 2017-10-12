from collections import OrderedDict

from dask import persist
from sklearn.preprocessing import imputation as skimputation
from daskml.utils import slice_columns
import dask.dataframe as dd
import numpy as np


class Imputer(skimputation.Imputer):
    def __init__(self, missing_values="NaN", strategy="mean",
                 axis=0, verbose=0, copy=True, columns=None):
        super().__init__(missing_values="NaN", strategy="mean",
                         axis=0, verbose=0, copy=True)
        self.columns = columns

        if (not copy or missing_values != "NaN" or strategy != "mean"
                or axis != 0 or verbose != 0):
            raise NotImplementedError()

    def fit(self, X, y=None):
        to_persist = OrderedDict()

        # Check parameters
        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.axis not in [0, 1]:
            raise ValueError("Can only impute missing values on axis 0 and 1, "
                             " got axis={0}".format(self.axis))

        _X = slice_columns(X, self.columns)
        if self.missing_values != "NaN":
            _X = _X.where(_X != self.missing_values, np.nan)

        if self.strategy == "mean":
            to_persist["statistics_"] = _X.mean()
        elif self.strategy == "median":
            to_persist["statistics_"] = _X.median()
        elif self.strategy == "most_frequent":
            raise NotImplementedError()

        values = persist(*to_persist.values())
        for k, v in zip(to_persist, values):
            setattr(self, k, v)

        return self

    def transform(self, X):
        _X = slice_columns(X, self.columns)
        if self.missing_values != "NaN":
            _X = _X.where(_X != self.missing_values, np.nan)
        if isinstance(_X, dd._Frame) and self.columns:
            for column in self.columns:
                X[column] = _X[column]
            return X.fillna(self.statistics_)
        else:
            return _X.fillna(self.statistics_)
