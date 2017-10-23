import dask_ml.preprocessing as dpp
import sklearn.preprocessing as spp
import dask.dataframe as dd
import numpy as np
import pytest
import pandas as pd
from dask_ml.datasets import make_classification
from dask.array.utils import assert_eq as assert_eq_ar
from dask import compute


X, y = make_classification(chunks=50)
X_with_zeros = X.copy()
X_with_zeros[X < 0] = 0
X_with_nan = X.copy()
X_with_nan[X < 0] = np.nan
df_with_zeros = X_with_zeros.to_dask_dataframe().rename(columns=str)
df_with_nan = df_with_zeros.mask(df_with_zeros > 1)


@pytest.mark.parametrize('X,missing_values,columns,strategy', [
    (X_with_nan, "NaN", None, "mean"),
    (X_with_nan, np.nan, None, "mean"),
    (X_with_zeros, 0, None, "mean"),
    (df_with_nan, "NaN", ['1', '2'], "mean"),
    (df_with_nan, np.nan, ['1', '2'], "mean"),
    (df_with_zeros, 0, ['1', '2'], "mean"),
    (df_with_zeros, 0, None, "mean"),
    (df_with_nan, "NaN", ['1', '2'], "median"),
    (df_with_nan, np.nan, ['1', '2'], "median"),
    (df_with_zeros, 0, ['1', '2'], "median"),
    (df_with_zeros, 0, None, "median")
])
class TestImputer(object):
    def test_fit(self, X, missing_values, columns, strategy):
        a = dpp.Imputer(columns=columns, missing_values=missing_values,
                        strategy=strategy)
        b = spp.Imputer(missing_values=missing_values, strategy=strategy)
        columns_ix = list(map(int, columns) if columns else
                          range(len(X.columns) if isinstance(X, dd._Frame)
                                else X.shape[1]))

        a.fit(X if strategy != "median" else (X.repartition(npartitions=1)
                                              if isinstance(X, dd._Frame)
                                              else X.rechunk(1)))
        b.fit(X.compute())

        c = a.statistics_.compute()

        assert_eq_ar(c, b.statistics_[columns_ix])

    def test_fit_transform(self, X, missing_values, columns, strategy):
        a = dpp.Imputer(columns=columns, missing_values=missing_values,
                        strategy=strategy)
        b = spp.Imputer(missing_values=missing_values, strategy=strategy)
        columns_ix = list(map(int, columns) if columns else
                          range(len(X.columns) if isinstance(X, dd._Frame)
                                else X.shape[1]))

        ta = a.fit_transform(X if strategy != "median"
                             else (X.repartition(npartitions=1)
                                   if isinstance(X, dd._Frame)
                                   else X.rechunk(1)))
        tb = b.fit_transform(X.compute())[:, columns_ix]

        assert_eq_ar(ta.values if isinstance(X, dd._Frame) else ta, tb)
