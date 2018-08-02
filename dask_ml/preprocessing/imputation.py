import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import sklearn.preprocessing

from ..utils import check_array


class Imputer(sklearn.preprocessing.Imputer):
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
            return super(Imputer, self).fit(X, y=y)

        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError(
                "Can only use these strategies: {0} "
                " got strategy={1}".format(allowed_strategies, self.strategy)
            )

        if self.axis != 0:
            raise ValueError(
                "Can only impute missing values on axis 0"
                " got axis={0}".format(self.axis)
            )

        if self.missing_values != "NaN":
            raise ValueError(
                "dask_ml.preprocessing.Imputer only supports 'missing_values=NaN'."
            )

        X = self._check_array(X)

        if isinstance(X, da.Array):
            self._fit_array(X)
        else:
            self._fit_frame(X)
        return self

    def _fit_array(self, X):
        if self.strategy != "mean":
            msg = (
                "Can only use strategy='mean' with Dask Array. "
                "Use 'mean' or convert 'X' to a Dask DataFrame."
            )
            raise ValueError(msg)

        avg = da.nanmean(X, axis=0).compute()
        self.statistics_ = avg

    def fit_frame(self, X):
        if self.strategy == "mean":
            avg = X.mean(axis=0).compute().values
        elif self.strategy == "median":
            avg = X.quantile().compute().values
        else:
            avg = np.concatenate(
                dask.compute(
                    *[X[col].value_counts().nlargest(1).values for col in X.columns]
                )
            )
        self.statistics_ = avg

    def transform(self, X):
        if isinstance(X, (pd.Series, dd.Series, dd.DataFrame)):
            return X.fillna(self.statistics_)

        elif isinstance(X, da.Array):
            return da.where(da.isnull(X), self.statistics_, X)
        else:
            return super(Imputer, self).transform(X)
