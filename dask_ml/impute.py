import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from packaging.version import parse

from ._compat import SK_VERSION
from .utils import check_array

if SK_VERSION >= parse("0.20.0.dev0"):
    import sklearn.impute
else:
    raise ImportError("dask_ml.impute is only available with scikit-learn>= 0.20")


class SimpleImputer(sklearn.impute.SimpleImputer):
    _types = (pd.Series, pd.DataFrame, dd.Series, dd.DataFrame, da.Array)

    def _check_array(self, X):
        return check_array(
            X,
            accept_dask_dataframe=True,
            accept_unknown_chunks=True,
            preserve_pandas_dataframe=True,
        )

    def fit(self, X, y=None):
        # Check parameters
        if not isinstance(X, self._types):
            return super(SimpleImputer, self).fit(X, y=y)

        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(
                "Can only use these strategies: {0} "
                " got strategy={1}".format(allowed_strategies, self.strategy)
            )

        if getattr(self, "axis", 0) != 0:
            raise ValueError(
                "Can only impute missing values on axis 0"
                " got axis={0}".format(self.axis)
            )

        if not np.isnan(self.missing_values):
            raise ValueError(
                "dask_ml.preprocessing.Imputer only supports 'missing_values=np.nan'."
            )

        X = self._check_array(X)

        if isinstance(X, da.Array):
            self._fit_array(X)
        else:
            self._fit_frame(X)
        return self

    def _fit_array(self, X):
        if self.strategy not in {"mean", "constant"}:
            msg = (
                "Can only use strategy='mean' with Dask Array. "
                "Use 'mean' or convert 'X' to a Dask DataFrame."
            )
            raise ValueError(msg)

        if self.strategy == "mean":
            statistics = da.nanmean(X, axis=0).compute()
        else:
            statistics = np.full(X.shape[1], self.strategy)

        self.statistics_, = da.compute(statistics)

    def _fit_frame(self, X):
        if self.strategy == "mean":
            avg = X.mean(axis=0).values
        elif self.strategy == "median":
            avg = X.quantile().values
        elif self.strategy == "constant":
            avg = np.full(len(X.columns), self.strategy)
        else:
            avg = np.concatenate(
                *[X[col].value_counts().nlargest(1).values for col in X.columns]
            )

        self.statistics_ = pd.Series(dask.compute(avg)[0], index=X.columns)

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame, dd.Series, dd.DataFrame)):
            return X.fillna(self.statistics_)

        elif isinstance(X, da.Array):
            return da.where(da.isnull(X), self.statistics_, X)
        else:
            return super(SimpleImputer, self).transform(X)
