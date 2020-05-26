import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import sklearn.impute

from .utils import check_array


class SimpleImputer(sklearn.impute.SimpleImputer):
    _types = (pd.Series, pd.DataFrame, dd.Series, dd.DataFrame, da.Array)

    def _check_array(self, X):
        return check_array(
            X,
            accept_dask_dataframe=True,
            accept_unknown_chunks=True,
            preserve_pandas_dataframe=True,
            force_all_finite=False,
        )

    def fit(self, X, y=None):
        # Check parameters
        if not isinstance(X, self._types):
            return super(SimpleImputer, self).fit(X, y=y)

        if hasattr(self, "add_indicator"):
            # scikit-learn version has add_indicator
            if self.add_indicator:
                msg = "dask-ml does not currently implement add_indicator" ""
                raise NotImplementedError(msg)
            self.indicator_ = None

        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(
                "Can only use these strategies: {0} "
                " got strategy={1}".format(allowed_strategies, self.strategy)
            )

        if not (pd.isna(self.missing_values) or self.strategy == "constant"):
            raise ValueError(
                "dask_ml.preprocessing.Imputer only supports non-NA values for "
                "'missing_values' when 'strategy=constant'."
            )

        X = self._check_array(X)

        if isinstance(X, da.Array):
            self._fit_array(X)
        else:
            self._fit_frame(X)
        self.n_features_in_ = X.shape[1]
        return self

    def _fit_array(self, X):
        if self.strategy not in {"mean", "constant"}:
            msg = "Can only use strategy='mean' or 'constant' with Dask Array."
            raise ValueError(msg)

        if self.strategy == "mean":
            statistics = da.nanmean(X, axis=0).compute()
        else:
            statistics = np.full(X.shape[1], self.fill_value, dtype=X.dtype)

        (self.statistics_,) = da.compute(statistics)

    def _fit_frame(self, X):
        if self.strategy == "mean":
            avg = X.mean(axis=0).values
        elif self.strategy == "median":
            avg = X.quantile().values
        elif self.strategy == "constant":
            avg = np.full(len(X.columns), self.fill_value)
        else:
            avg = [X[col].value_counts().nlargest(1).index for col in X.columns]
            avg = np.concatenate(*dask.compute(avg))

        self.statistics_ = pd.Series(dask.compute(avg)[0], index=X.columns)

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame, dd.Series, dd.DataFrame)):
            return X.fillna(self.statistics_)

        elif isinstance(X, da.Array):
            return da.where(da.isnull(X), self.statistics_, X)
        else:
            return super(SimpleImputer, self).transform(X)
