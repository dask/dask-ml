import pytest
import sklearn.preprocessing as spp

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.array.utils import assert_eq as assert_eq_ar
from dask.array.utils import assert_eq as assert_eq_df

import daskml.preprocessing as dpp
from daskml.datasets import make_classification
from daskml.utils import assert_estimator_equal


X, y = make_classification(chunks=2)
df = X.to_dask_dataframe().rename(columns=str)
df2 = dd.from_pandas(pd.DataFrame(5*[range(42)]).T.rename(columns=str),
                     npartitions=5)


class TestStandardScaler(object):
    def test_basic(self):
        a = dpp.StandardScaler()
        b = spp.StandardScaler()

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a, b)

    def test_inverse_transform(self):
        a = dpp.StandardScaler()
        assert_eq_ar(a.inverse_transform(a.fit_transform(X)).compute(),
                     X.compute())


class TestMinMaxScaler(object):
    def test_basic(self):
        a = dpp.MinMaxScaler()
        b = spp.MinMaxScaler()

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a, b, exclude='n_samples_seen_')

    def test_inverse_transform(self):
        a = dpp.MinMaxScaler()
        assert_eq_ar(a.inverse_transform(a.fit_transform(X)).compute(),
                     X.compute())

    def test_df_inverse_transform(self):
        mask = ["3", "4"]
        a = dpp.MinMaxScaler(columns=mask)
        assert_eq_df(a.inverse_transform(a.fit_transform(df2)).compute(),
                     df2.compute())

    def test_df_values(self):
        a = dpp.MinMaxScaler()
        assert_eq_ar(a.fit_transform(X).compute(),
                     a.fit_transform(df).compute().as_matrix())

    def test_df_column_slice(self):
        mask = ["3", "4"]
        mask_ix = [mask.index(x) for x in mask]
        a = dpp.MinMaxScaler(columns=mask)
        b = spp.MinMaxScaler()

        dfa = a.fit_transform(df2).compute()
        mxb = b.fit_transform(df2.compute())

        assert isinstance(dfa, pd.DataFrame)
        assert_eq_df(dfa[mask].as_matrix(), mxb[:, mask_ix])
        assert_eq_df(dfa.drop(mask, axis=1),
                     df2.drop(mask, axis=1).compute())


class TestQuantileTransformer(object):

    def test_basic(self):
        rs = da.random.RandomState(0)
        a = dpp.QuantileTransformer()
        b = spp.QuantileTransformer()

        X = rs.uniform(size=(100, 3), chunks=50)
        a.fit(X)
        b.fit(X)
        assert_estimator_equal(a, b, atol=.02)

        # set the quantiles, so that from here out, we're exact
        a.quantiles_ = b.quantiles_
        assert_eq_ar(a.transform(X), b.transform(X))
        assert_eq_ar(X, a.inverse_transform(a.transform(X)))

    @pytest.mark.parametrize('type_, kwargs', [
        (np.array, {}),
        (da.from_array, {'chunks': 10}),
        (pd.DataFrame, {'columns': ['a', 'b', 'c']}),
        (dd.from_array, {"columns": ['a', 'b', 'c']}),
    ]
    )
    def test_types(self, type_, kwargs):
        X = np.random.uniform(size=(20, 3))
        dX = type_(X, **kwargs)
        qt = spp.QuantileTransformer()
        qt.fit(X)
        dqt = dpp.QuantileTransformer()
        dqt.fit(dX)
