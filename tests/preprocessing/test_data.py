import pytest
import sklearn.preprocessing as spp
from sklearn.exceptions import NotFittedError

import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
import numpy as np
import pandas as pd
import pandas.util.testing as tm
from pandas.api.types import is_categorical_dtype, is_object_dtype
from dask import compute
from dask.array.utils import assert_eq as assert_eq_ar
from dask.array.utils import assert_eq as assert_eq_df

import dask_ml.preprocessing as dpp
from dask_ml.datasets import make_classification
from dask_ml.utils import assert_estimator_equal


X, y = make_classification(chunks=2)
df = X.to_dask_dataframe().rename(columns=str)
df2 = dd.from_pandas(pd.DataFrame(5 * [range(42)]).T.rename(columns=str),
                     npartitions=5)
raw = pd.DataFrame({"A": ['a', 'b', 'c', 'a'],
                    "B": ['a', 'b', 'c', 'a'],
                    "C": ['a', 'b', 'c', 'a'],
                    "D": [1, 2, 3, 4]},
                   columns=['A', 'B', 'C', 'D'])
dummy = pd.DataFrame({"A": pd.Categorical(['a', 'b', 'c', 'a'], ordered=True),
                      "B": pd.Categorical(['a', 'b', 'c', 'a'], ordered=False),
                      "C": pd.Categorical(['a', 'b', 'c', 'a'],
                                          categories=['a', 'b', 'c', 'd']),
                      "D": [1, 2, 3, 4]},
                     columns=['A', 'B', 'C', 'D'])


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


class TestCategorizer(object):

    def test_ce(self):
        ce = dpp.Categorizer()
        original = raw.copy()
        trn = ce.fit_transform(raw)
        assert is_categorical_dtype(trn['A'])
        assert is_categorical_dtype(trn['B'])
        assert is_categorical_dtype(trn['C'])
        assert trn['D'].dtype == int
        tm.assert_index_equal(ce.columns_, pd.Index(['A', 'B', 'C']))
        tm.assert_frame_equal(raw, original)

    def test_given_categories(self):
        cats = ['a', 'b', 'c', 'd']
        ce = dpp.Categorizer(categories={'A': (cats, True)})
        trn = ce.fit_transform(raw)
        assert trn['A'].dtype == 'category'
        tm.assert_index_equal(trn['A'].cat.categories, pd.Index(cats))
        assert all(trn['A'].cat.categories == cats)
        assert trn['A'].cat.ordered

    def test_dask(self):
        a = dd.from_pandas(raw, npartitions=2)
        ce = dpp.Categorizer()
        trn = ce.fit_transform(a)
        assert is_categorical_dtype(trn['A'])
        assert is_categorical_dtype(trn['B'])
        assert is_categorical_dtype(trn['C'])
        assert trn['D'].dtype == int
        tm.assert_index_equal(ce.columns_, pd.Index(['A', 'B', 'C']))

    def test_columns(self):
        ce = dpp.Categorizer(columns=['A'])
        trn = ce.fit_transform(raw)
        assert is_categorical_dtype(trn['A'])
        assert is_object_dtype(trn['B'])

    @pytest.mark.skipif(dpp.data._HAS_CTD, reason="No CategoricalDtypes")
    def test_non_categorical_dtype(self):
        ce = dpp.Categorizer()
        ce.fit(raw)
        idx, ordered = ce.categories_['A']
        tm.assert_index_equal(idx, pd.Index(['a', 'b', 'c']))
        assert ordered is False

    @pytest.mark.skipif(not dpp.data._HAS_CTD, reason="Has CategoricalDtypes")
    def test_categorical_dtype(self):
        ce = dpp.Categorizer()
        ce.fit(raw)
        assert (hash(ce.categories_['A']) ==
                hash(pd.api.types.CategoricalDtype(['a', 'b', 'c'], False)))

    def test_raises(self):
        ce = dpp.Categorizer()
        X = np.array([[0, 0], [1, 1]])
        with pytest.raises(TypeError):
            ce.fit(X)

        X = da.from_array(X, chunks=(2, 2))
        with pytest.raises(TypeError):
            ce.fit(X)

        with pytest.raises(NotFittedError):
            ce.transform(raw)


class TestDummyEncoder:

    @pytest.mark.parametrize('daskify', [False, True])
    @pytest.mark.parametrize('values', [True, False])
    def test_basic(self, daskify, values):
        de = dpp.DummyEncoder()
        df = dummy[['A', 'D']]
        if daskify:
            df = dd.from_pandas(df, 2)
        de = de.fit(df)
        trn = de.transform(df)

        expected = pd.DataFrame({
            "D": np.array([1, 2, 3, 4]),
            "A_a": np.array([1, 0, 0, 1], dtype='uint8'),
            "A_b": np.array([0, 1, 0, 0], dtype='uint8'),
            "A_c": np.array([0, 0, 1, 0], dtype='uint8'),
        }, columns=['D', 'A_a', 'A_b', 'A_c'])

        assert_eq(trn, expected)

        if values:
            trn = trn.values

        result = de.inverse_transform(trn)

        if daskify:
            df = df.compute()
            result = result.compute()

        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("daskify", [False, True])
    def test_drop_first(self, daskify):
        if daskify:
            df = dd.from_pandas(dummy, 2)
        else:
            df = dummy
        de = dpp.DummyEncoder(drop_first=True)
        trn = de.fit_transform(df)
        assert len(trn.columns) == 8

        result = de.inverse_transform(trn)
        if daskify:
            result, df = compute(result, df)
        tm.assert_frame_equal(result, dummy)

    def test_da(self):
        a = dd.from_pandas(dummy, npartitions=2)
        de = dpp.DummyEncoder()
        result = de.fit_transform(a)
        assert isinstance(result, dd.DataFrame)

    def test_transform_raises(self):
        de = dpp.DummyEncoder()
        de.fit(dummy)
        with pytest.raises(ValueError) as rec:
            de.transform(dummy.drop("B", axis='columns'))
        assert rec.match("Columns of 'X' do not match the training")
