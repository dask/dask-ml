from collections import OrderedDict

from dask import persist
from sklearn.preprocessing import imputation as skimputation
from daskml.utils import slice_columns
import dask.dataframe as dd


class Imputer(skimputation.Imputer):
    def __init__(self, missing_values="NaN", strategy="mean",
                 axis=0, verbose=0, copy=True, columns=None):
        super().__init__(missing_values="NaN", strategy="mean",
                         axis=0, verbose=0, copy=True)
        self._columns = columns

        if (not copy or missing_values != "NaN" or strategy != "mean"
                or axis != 0 or verbose != 0):
            raise NotImplementedError()

    def fit(self, X, y=None):
        self._cache = dict()
        to_persist = OrderedDict()
        if not isinstance(X, dd.DataFrame):
            raise NotImplementedError()

        # Check parameters
        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.axis not in [0, 1]:
            raise ValueError("Can only impute missing values on axis 0 and 1, "
                             " got axis={0}".format(self.axis))

        _X = slice_columns(X, self._columns)
        if self.strategy == "mean":
            to_persist["statistics_"] = X[list(_X.columns)].mean()

        values = persist(*to_persist.values(), cache=self._cache)
        for k, v in zip(to_persist, values):
            setattr(self, k, v)

        return self

    def transform(self, X):
        return X.fillna(self.statistics_)
