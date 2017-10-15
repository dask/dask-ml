import dask_ml.preprocessing as dpp
import sklearn.preprocessing as spp
import dask.dataframe as dd
import numpy as np
import pytest
from dask_ml.datasets import make_classification
from dask.array.utils import assert_eq as assert_eq_ar


X, y = make_classification(chunks=2)
X_with_zeros = X.copy()
X_with_zeros[X < 0] = 0
X_with_nan = X.copy()
X_with_nan[X < 0] = np.nan
df_with_zeros = X_with_zeros.to_dask_dataframe().rename(columns=str)
df_with_nan = df_with_zeros.mask(df_with_zeros > 1)


@pytest.mark.parametrize('strategy', ["mean"])
@pytest.mark.parametrize('X,missing_values,columns', [
    (X_with_nan, "NaN", None),
    (X_with_nan, np.nan, None),
    (X_with_zeros, 0, None),
    (df_with_nan, "NaN", ['1', '2']),
    (df_with_nan, np.nan, ['1', '2']),
    (df_with_zeros, 0, ['1', '2']),
    (df_with_zeros, 0, None)
])
class TestImputer(object):
    def test_fit(self, X, missing_values, columns, strategy):
        a = dpp.Imputer(columns=columns, missing_values=missing_values,
                        strategy=strategy)
        b = spp.Imputer(missing_values=missing_values, strategy=strategy)
        columns_ix = list(map(int, columns) if columns else
                          range(len(X.columns) if isinstance(X, dd._Frame)
                                else X.shape[1]))

        if strategy == "median":
            X = X.repartition(npartitions=1)

        a.fit(X)
        b.fit(X.compute())
        assert_eq_ar(a.statistics_.compute(),
                     b.statistics_[columns_ix])

    def test_fit_transform(self, X, missing_values, columns, strategy):
        a = dpp.Imputer(columns=columns, missing_values=missing_values,
                        strategy=strategy)
        b = spp.Imputer(missing_values=missing_values, strategy=strategy)
        columns_ix = list(map(int, columns) if columns else
                          range(len(X.columns) if isinstance(X, dd._Frame)
                                else X.shape[1]))
        if strategy == "median":
            X = X.repartition(npartitions=1)

        ta = a.fit_transform(X).compute()
        tb = b.fit_transform(X.compute())[:, columns_ix]
        if isinstance(X, dd._Frame):
            assert_eq_ar(ta.values, tb)
        else:
            assert_eq_ar(ta, tb)
