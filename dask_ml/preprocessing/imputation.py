from collections import OrderedDict

from dask import persist
from sklearn.preprocessing import imputation as skimputation
from daskml.utils import slice_columns
import dask.dataframe as dd
import numpy as np



def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


class Imputer(skimputation.Imputer):
    def __init__(self, missing_values="NaN", strategy="mean",
                 axis=0, verbose=0, copy=True, columns=None):
        super().__init__(missing_values="NaN", strategy="mean",
                         axis=0, verbose=0, copy=True)
        self.columns = columns
        self.missing_values = missing_values
        self.strategy = strategy
        self.axis = axis
        self.verbose = verbose
        self.copy = copy

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

        values = persist(*to_persist.values())
        for k, v in zip(to_persist, values):
            setattr(self, k, v)

        return self

    def transform(self, X):
        pass
