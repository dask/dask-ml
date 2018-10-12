import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pytest
import sklearn.preprocessing as spp
from dask import compute
from dask.array.utils import assert_eq as assert_eq_ar
from dask.dataframe.utils import assert_eq as assert_eq_df
from pandas.api.types import is_categorical_dtype, is_object_dtype
from sklearn.exceptions import NotFittedError

import dask_ml.preprocessing as dpp
from dask_ml.datasets import make_classification
from dask_ml.utils import assert_estimator_equal

X, y = make_classification(chunks=50)
df = X.to_dask_dataframe().rename(columns=str)
df2 = dd.from_pandas(pd.DataFrame(5 * [range(42)]).T.rename(columns=str), npartitions=5)
raw = pd.DataFrame(
    {
        "A": ["a", "b", "c", "a"],
        "B": ["a", "b", "c", "a"],
        "C": ["a", "b", "c", "a"],
        "D": [1, 2, 3, 4],
    },
    columns=["A", "B", "C", "D"],
)
dummy = pd.DataFrame(
    {
        "A": pd.Categorical(["a", "b", "c", "a"], ordered=True),
        "B": pd.Categorical(["a", "b", "c", "a"], ordered=False),
        "C": pd.Categorical(["a", "b", "c", "a"], categories=["a", "b", "c", "d"]),
        "D": [1, 2, 3, 4],
    },
    columns=["A", "B", "C", "D"],
)


@pytest.fixture
def pandas_df():
    return pd.DataFrame(5 * [range(42)]).T.rename(columns=str)


@pytest.fixture
def dask_df(pandas_df):
    return dd.from_pandas(pandas_df, npartitions=5)


class TestStandardScaler(object):
    def test_basic(self):
        a = dpp.StandardScaler()
        b = spp.StandardScaler()

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a, b, exclude="n_samples_seen_")

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.DataConversionWarning")
    def test_input_types(self, dask_df, pandas_df):
        a = dpp.StandardScaler()
        b = spp.StandardScaler()

        assert_estimator_equal(
            a.fit(dask_df.values), a.fit(dask_df), exclude="n_samples_seen_"
        )

        assert_estimator_equal(
            a.fit(dask_df), b.fit(pandas_df), exclude="n_samples_seen_"
        )

        assert_estimator_equal(
            a.fit(dask_df.values), b.fit(pandas_df), exclude="n_samples_seen_"
        )

        assert_estimator_equal(
            a.fit(dask_df), b.fit(pandas_df.values), exclude="n_samples_seen_"
        )

        assert_estimator_equal(
            a.fit(dask_df.values), b.fit(pandas_df.values), exclude="n_samples_seen_"
        )

    def test_inverse_transform(self):
        a = dpp.StandardScaler()
        result = a.inverse_transform(a.fit_transform(X))
        assert dask.is_dask_collection(result)
        assert_eq_ar(result, X)


class TestMinMaxScaler(object):
    def test_basic(self):
        a = dpp.MinMaxScaler()
        b = spp.MinMaxScaler()

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a, b, exclude="n_samples_seen_")

    def test_inverse_transform(self):
        a = dpp.MinMaxScaler()
        result = a.inverse_transform(a.fit_transform(X))
        assert dask.is_dask_collection(result)
        assert_eq_ar(result, X)

    @pytest.mark.xfail(reason="removed columns")
    def test_df_inverse_transform(self):
        mask = ["3", "4"]
        a = dpp.MinMaxScaler(columns=mask)
        result = a.inverse_transform(a.fit_transform(df2))
        assert dask.is_dask_colelction(result)
        assert_eq_df(result, df2)

    def test_df_values(self):
        est1 = dpp.MinMaxScaler()
        est2 = dpp.MinMaxScaler()

        result_ar = est1.fit_transform(X)
        result_df = est2.fit_transform(df)

        for attr in ["data_min_", "data_max_", "data_range_", "scale_", "min_"]:
            assert_eq_ar(getattr(est1, attr), getattr(est2, attr).values)

        assert_eq_ar(est1.transform(X), est2.transform(X))
        assert_eq_ar(est1.transform(df).values, est2.transform(X))
        assert_eq_ar(est1.transform(X), est2.transform(df).values)

        if hasattr(result_df, "values"):
            result_df = result_df.values
        assert_eq_ar(result_ar, result_df)

    @pytest.mark.xfail(reason="removed columns")
    def test_df_column_slice(self):
        mask = ["3", "4"]
        mask_ix = [mask.index(x) for x in mask]
        a = dpp.MinMaxScaler(columns=mask)
        b = spp.MinMaxScaler()

        dfa = a.fit_transform(df2)
        mxb = b.fit_transform(df2.compute())

        assert isinstance(dfa, dd.DataFrame)
        assert_eq_ar(dfa[mask].values, mxb[:, mask_ix])
        assert_eq_df(dfa.drop(mask, axis=1), df2.drop(mask, axis=1))


class TestRobustScaler(object):
    def test_fit(self):
        a = dpp.RobustScaler()
        b = spp.RobustScaler()

        # bigger data to make percentile more reliable
        # and not centered around 0 to make rtol work
        X, y = make_classification(n_samples=1000, chunks=200, random_state=0)
        X = X + 3

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a, b, rtol=0.2)

    def test_transform(self):
        a = dpp.RobustScaler()
        b = spp.RobustScaler()

        a.fit(X)
        b.fit(X.compute())

        # overwriting dask-ml's fitted attributes to have them exactly equal
        # (the approximate equality is tested above)
        a.scale_ = b.scale_
        a.center_ = b.center_

        assert dask.is_dask_collection(a.transform(X))
        assert_eq_ar(a.transform(X), b.transform(X.compute()))

    def test_inverse_transform(self):
        a = dpp.RobustScaler()
        result = a.inverse_transform(a.fit_transform(X))
        assert dask.is_dask_collection(result)
        assert_eq_ar(result, X)

    def test_df_values(self):
        est1 = dpp.RobustScaler()
        est2 = dpp.RobustScaler()

        result_ar = est1.fit_transform(X)
        result_df = est2.fit_transform(df)
        if hasattr(result_df, "values"):
            result_df = result_df.values
        assert_eq_ar(result_ar, result_df)

        for attr in ["scale_", "center_"]:
            assert_eq_ar(getattr(est1, attr), getattr(est2, attr))

        assert_eq_ar(est1.transform(X), est2.transform(X))
        assert_eq_ar(est1.transform(df).values, est2.transform(X))
        assert_eq_ar(est1.transform(X), est2.transform(df).values)

        # different data types
        df["0"] = df["0"].astype("float32")
        result_ar = est1.fit_transform(X)
        result_df = est2.fit_transform(df)
        if hasattr(result_df, "values"):
            result_df = result_df.values
        assert_eq_ar(result_ar, result_df)


class TestQuantileTransformer(object):
    def test_basic(self):
        rs = da.random.RandomState(0)
        a = dpp.QuantileTransformer()
        b = spp.QuantileTransformer()

        X = rs.uniform(size=(100, 3), chunks=50)
        a.fit(X)
        b.fit(X)
        assert_estimator_equal(a, b, atol=0.02)

        # set the quantiles, so that from here out, we're exact
        a.quantiles_ = b.quantiles_
        assert_eq_ar(a.transform(X), b.transform(X))
        assert_eq_ar(X, a.inverse_transform(a.transform(X)))

    @pytest.mark.parametrize(
        "type_, kwargs",
        [
            (np.array, {}),
            (da.from_array, {"chunks": 10}),
            (pd.DataFrame, {"columns": ["a", "b", "c"]}),
            (dd.from_array, {"columns": ["a", "b", "c"]}),
        ],
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
        assert is_categorical_dtype(trn["A"])
        assert is_categorical_dtype(trn["B"])
        assert is_categorical_dtype(trn["C"])
        assert trn["D"].dtype == int
        tm.assert_index_equal(ce.columns_, pd.Index(["A", "B", "C"]))
        tm.assert_frame_equal(raw, original)

    def test_given_categories(self):
        cats = ["a", "b", "c", "d"]
        ce = dpp.Categorizer(categories={"A": (cats, True)})
        trn = ce.fit_transform(raw)
        assert trn["A"].dtype == "category"
        tm.assert_index_equal(trn["A"].cat.categories, pd.Index(cats))
        assert all(trn["A"].cat.categories == cats)
        assert trn["A"].cat.ordered

    def test_dask(self):
        a = dd.from_pandas(raw, npartitions=2)
        ce = dpp.Categorizer()
        trn = ce.fit_transform(a)
        assert is_categorical_dtype(trn["A"])
        assert is_categorical_dtype(trn["B"])
        assert is_categorical_dtype(trn["C"])
        assert trn["D"].dtype == int
        tm.assert_index_equal(ce.columns_, pd.Index(["A", "B", "C"]))

    def test_columns(self):
        ce = dpp.Categorizer(columns=["A"])
        trn = ce.fit_transform(raw)
        assert is_categorical_dtype(trn["A"])
        assert is_object_dtype(trn["B"])

    @pytest.mark.skipif(dpp.data._HAS_CTD, reason="No CategoricalDtypes")
    def test_non_categorical_dtype(self):
        ce = dpp.Categorizer()
        ce.fit(raw)
        idx, ordered = ce.categories_["A"]
        tm.assert_index_equal(idx, pd.Index(["a", "b", "c"]))
        assert ordered is False

    @pytest.mark.skipif(not dpp.data._HAS_CTD, reason="Has CategoricalDtypes")
    def test_categorical_dtype(self):
        ce = dpp.Categorizer()
        ce.fit(raw)
        assert hash(ce.categories_["A"]) == hash(
            pd.api.types.CategoricalDtype(["a", "b", "c"], False)
        )

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
    @pytest.mark.parametrize("daskify", [False, True])
    @pytest.mark.parametrize("values", [True, False])
    def test_basic(self, daskify, values):
        de = dpp.DummyEncoder()
        df = dummy[["A", "D"]]
        if daskify:
            df = dd.from_pandas(df, 2)
        de = de.fit(df)
        trn = de.transform(df)

        expected = pd.DataFrame(
            {
                "D": np.array([1, 2, 3, 4]),
                "A_a": np.array([1, 0, 0, 1], dtype="uint8"),
                "A_b": np.array([0, 1, 0, 0], dtype="uint8"),
                "A_c": np.array([0, 0, 1, 0], dtype="uint8"),
            },
            columns=["D", "A_a", "A_b", "A_c"],
        )

        assert_eq_df(trn, expected)

        if values:
            trn = trn.values

        result = de.inverse_transform(trn)

        if daskify:
            df = df.compute()
            result = result.compute()

        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("daskify", [False, True])
    def test_encode_subset_of_columns(self, daskify):
        de = dpp.DummyEncoder(columns=["B"])
        df = dummy[["A", "B"]]
        if daskify:
            df = dd.from_pandas(df, 2)
        de = de.fit(df)
        trn = de.transform(df)

        expected = pd.DataFrame(
            {
                "A": pd.Categorical(["a", "b", "c", "a"], ordered=True),
                "B_a": np.array([1, 0, 0, 1], dtype="uint8"),
                "B_b": np.array([0, 1, 0, 0], dtype="uint8"),
                "B_c": np.array([0, 0, 1, 0], dtype="uint8"),
            },
            columns=["A", "B_a", "B_b", "B_c"],
        )

        assert_eq_df(trn, expected)

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
            de.transform(dummy.drop("B", axis="columns"))
        assert rec.match("Columns of 'X' do not match the training")

    def test_inverse_transform(self):
        de = dpp.DummyEncoder()
        df = dd.from_pandas(
            pd.DataFrame(
                {"A": np.arange(10), "B": pd.Categorical(["a"] * 4 + ["b"] * 6)}
            ),
            npartitions=2,
        )
        de.fit(df)
        assert_eq_df(df, de.inverse_transform(de.transform(df)))
        assert_eq_df(df, de.inverse_transform(de.transform(df).values))


class TestOrdinalEncoder:
    @pytest.mark.parametrize("daskify", [False, True])
    @pytest.mark.parametrize("values", [True, False])
    def test_basic(self, daskify, values):
        de = dpp.OrdinalEncoder()
        df = dummy[["A", "D"]]
        if daskify:
            df = dd.from_pandas(df, 2)
        de = de.fit(df)
        trn = de.transform(df)

        expected = pd.DataFrame(
            {"A": np.array([0, 1, 2, 0], dtype="int8"), "D": np.array([1, 2, 3, 4])},
            columns=["A", "D"],
        )

        assert_eq_df(trn, expected)

        if values:
            trn = trn.values

        result = de.inverse_transform(trn)

        if daskify:
            df = df.compute()
            result = result.compute()

        tm.assert_frame_equal(result, df)

    def test_da(self):
        a = dd.from_pandas(dummy, npartitions=2)
        de = dpp.OrdinalEncoder()
        result = de.fit_transform(a)
        assert isinstance(result, dd.DataFrame)

    def test_transform_raises(self):
        de = dpp.OrdinalEncoder()
        de.fit(dummy)
        with pytest.raises(ValueError) as rec:
            de.transform(dummy.drop("B", axis="columns"))
        assert rec.match("Columns of 'X' do not match the training")

    def test_inverse_transform(self):
        enc = dpp.OrdinalEncoder()
        df = dd.from_pandas(
            pd.DataFrame(
                {"A": np.arange(10), "B": pd.Categorical(["a"] * 4 + ["b"] * 6)}
            ),
            npartitions=2,
        )
        enc.fit(df)

        assert dask.is_dask_collection(enc.inverse_transform(enc.transform(df).values))
        assert dask.is_dask_collection(enc.inverse_transform(enc.transform(df)))

        assert_eq_df(df, enc.inverse_transform(enc.transform(df)))
        assert_eq_df(df, enc.inverse_transform(enc.transform(df)))
        assert_eq_df(df, enc.inverse_transform(enc.transform(df).values))
        assert_eq_df(df, enc.inverse_transform(enc.transform(df).values))


class TestPolynomialFeatures:
    def test_basic(self):
        a = dpp.PolynomialFeatures()
        b = spp.PolynomialFeatures()

        a.fit(X)
        b.fit(X.compute())
        assert_estimator_equal(a._transformer, b)

    def test_input_types(self):
        a = dpp.PolynomialFeatures()
        b = spp.PolynomialFeatures()

        assert_estimator_equal(a.fit(df), a.fit(df.compute()))
        assert_estimator_equal(a.fit(df), a.fit(df.compute().values))
        assert_estimator_equal(a.fit(df.values), a.fit(df.compute().values))
        assert_estimator_equal(a.fit(df), b.fit(df.compute()))
        assert_estimator_equal(a.fit(df), b.fit(df.compute().values))

    def test_estimator_comparison(self):
        a = dpp.PolynomialFeatures()
        b = spp.PolynomialFeatures()

        a.fit_transform(X).compute()
        b.fit_transform(X.compute())
        assert_estimator_equal(a, b)

    def test_array_transform(self):
        a = dpp.PolynomialFeatures()
        b = spp.PolynomialFeatures()

        res_a = a.fit_transform(X).compute()
        res_b = b.fit_transform(X.compute())
        assert_eq_ar(res_a, res_b)

    def test_transform_array(self):
        a = dpp.PolynomialFeatures()
        b = spp.PolynomialFeatures()

        # pass numpy array to fit_transform
        res_a1 = a.fit_transform(X.compute())
        # pass dask array to fit_transform
        res_a2 = a.fit_transform(X).compute()
        res_b = b.fit_transform(X.compute())
        assert_eq_ar(res_a1, res_b)
        assert_eq_ar(res_a2, res_b)

    def test_df_transform(self):
        a = dpp.PolynomialFeatures()
        b = spp.PolynomialFeatures()

        # pandas dataframe
        res_pdf = a.fit_transform(df.compute()).values
        # spp returns a numpy array
        res_b = b.fit_transform(df.compute())
        assert_eq_ar(res_pdf, res_b)

    def test_daskdf_transform(self):
        a = dpp.PolynomialFeatures()
        b = spp.PolynomialFeatures()

        # dask dataframe
        trans_dask_df = a.fit_transform(df)

        # pandas dataframe
        res_pandas_df = a.fit_transform(df.compute())
        # spp returns a numpy array
        res_b = b.fit_transform(df.compute())
        assert_eq_ar(trans_dask_df.compute().values, res_b)
        assert_eq_df(trans_dask_df.compute(), res_pandas_df)
