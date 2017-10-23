from dask_ml.utils import slice_columns
from sklearn.preprocessing import imputation as skimputation
import dask.dataframe as dd
import dask.array as da
import numpy as np


def _safe_eq(X, v):
    if isinstance(X, dd._Frame):
        return X.isnull() if v == "NaN" or np.isnan(v) else X == v
    else:
        return da.isnull(X) if v == "NaN" or np.isnan(v) else X == v


def _mask_values(X, missing_values):
    """Compute the masked version of X."""
    if missing_values == "NaN" or np.isnan(missing_values):
        return X if isinstance(X, dd._Frame) else da.ma.masked_invalid(X)
    else:
        return (X.mask(X == missing_values) if isinstance(X, dd._Frame)
                else da.ma.masked_equal(X, missing_values))


def _fit_columns_df(df, columns, estimator):
    df = df.copy()
    return {c: estimator(df[c]) for c in columns}


def _fit_columns_ar(ar, estimator):
    raise NotImplementedError()


class Imputer(skimputation.Imputer):
    def __init__(self, missing_values="NaN", strategy="mean",
                 axis=0, verbose=0, copy=True, columns=None):
        super().__init__(missing_values=missing_values, strategy=strategy,
                         axis=axis, verbose=verbose, copy=copy)
        self.columns = columns
        if not copy or axis != 0 or verbose != 0:
            raise NotImplementedError()

    def fit(self, X, y=None):
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
        _masked_X = _mask_values(_X, self.missing_values)
        self.statistics_ = self._dense_fit(_masked_X,
                                           self.strategy,
                                           self.missing_values).compute()

        return self

    def _dense_fit(self, _masked_X, strategy, missing_values):
        columns = (self.columns
                   if self.columns or isinstance(_masked_X, da.Array)
                   else _masked_X.columns)
        is_frame = isinstance(_masked_X, dd.DataFrame)

        if strategy == "mean":
            if is_frame:
                return _masked_X.mean()
            else:
                return da.ma.filled(_masked_X.mean(0))
        elif strategy == "median":
            if is_frame:
                return _masked_X.quantile(0.5)
            else:
                raise NotImplementedError("strategy='median' is not supported "
                                          "arrays. Use `dask.dataframe` or a "
                                          "different strategy instead.")
        else:
            raise NotImplementedError("Invalid strategy {}".format(strategy))

    def _sparse_fit(self):
        raise NotImplementedError()

    def transform(self, X):
        _X = slice_columns(X, self.columns).copy()
        is_frame = isinstance(X, dd.DataFrame)

        s = self.statistics_

        if is_frame:
            for c in _X.columns:
                _X[c] = _X[c].mask(_safe_eq(_X[c], self.missing_values), s[c])
        else:
            _X = da.vstack([da.where(_safe_eq(x, self.missing_values), s[i], x)
                            for i, x in enumerate(X.T)]).T
        return _X
