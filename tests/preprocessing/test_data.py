from daskml.datasets import make_classification
from sklearn.preprocessing import StandardScaler as StandardScaler_
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_
from daskml.preprocessing import StandardScaler
from daskml.preprocessing import MinMaxScaler

import dask.dataframe as dd
import pandas as pd
from dask.array.utils import assert_eq as assert_eq_ar
from dask.array.utils import assert_eq as assert_eq_df

from ..test_utils import assert_estimator_equal


X, y = make_classification(chunks=2)
df = X.to_dask_dataframe().rename(columns=str)
df2 = dd.from_pandas(pd.DataFrame(5*[range(42)]).T.rename(columns=str),
                     npartitions=5)


class TestStandardScaler(object):
    def test_basic(self):
        a = StandardScaler()
        b = StandardScaler_()

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a, b)

    def test_inverse_transform(self):
        a = StandardScaler()
        assert_eq_ar(a.inverse_transform(a.fit_transform(X)).compute(),
                     X.compute())


class TestMinMaxScaler(object):
    def test_basic(self):
        a = MinMaxScaler()
        b = MinMaxScaler_()

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a, b, exclude='n_samples_seen_')

    def test_inverse_transform(self):
        a = MinMaxScaler()
        assert_eq_ar(a.inverse_transform(a.fit_transform(X)).compute(),
                     X.compute())

    def test_df_inverse_transform(self):
        mask = ["3", "4"]
        a = MinMaxScaler(columns=mask)
        assert_eq_df(a.inverse_transform(a.fit_transform(df2)).compute(),
                     df2.compute())

    def test_df_values(self):
        a = MinMaxScaler()
        assert_eq_ar(a.fit_transform(X).compute(),
                     a.fit_transform(df).compute().as_matrix())

    def test_df_column_slice(self):
        mask = ["3", "4"]
        mask_ix = [mask.index(x) for x in mask]
        a = MinMaxScaler(columns=mask)
        b = MinMaxScaler_()

        dfa = a.fit_transform(df2).compute()
        mxb = b.fit_transform(df2.compute())

        assert isinstance(dfa, pd.DataFrame)
        assert_eq_df(dfa[mask].as_matrix(), mxb[:, mask_ix])
        assert_eq_df(dfa.drop(mask, axis=1),
                     df2.drop(mask, axis=1).compute())
