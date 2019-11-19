import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing as spp
from dask.array.utils import assert_eq as assert_eq_ar

import dask_ml.preprocessing as dpp
from dask_ml.utils import assert_estimator_equal

choices = np.array(["a", "b", "c"], dtype=str)
np_y = np.random.choice(choices, 100)
y = da.from_array(np_y, chunks=13)
s = dd.from_array(y)


@pytest.fixture
def pandas_series():
    y = np.random.choice(["a", "b", "c"], 100)
    return pd.Series(y)


@pytest.fixture
def dask_array(pandas_series):
    return da.from_array(pandas_series, chunks=5)


class TestLabelEncoder:
    def test_basic(self):
        a = dpp.LabelEncoder()
        b = spp.LabelEncoder()

        a.fit(y)
        b.fit(y.compute())
        exclude = {"dtype_"}
        assert_estimator_equal(a, b, exclude=exclude)

    def test_input_types(self, dask_array, pandas_series):
        a = dpp.LabelEncoder()
        b = spp.LabelEncoder()

        exclude = {"dtype_"}
        assert_estimator_equal(a.fit(dask_array), b.fit(pandas_series), exclude=exclude)

        assert_estimator_equal(
            a.fit(pandas_series), b.fit(pandas_series), exclude=exclude
        )

        assert_estimator_equal(
            a.fit(pandas_series.values), b.fit(pandas_series), exclude=exclude
        )

        assert_estimator_equal(
            a.fit(dask_array), b.fit(pandas_series.values), exclude=exclude
        )

    @pytest.mark.parametrize(
        "array",
        [
            y,
            pytest.param(
                s,
                marks=[
                    pytest.mark.xfail(reason="Incorrect 32-bit dtype.", strict=False)
                ],
            ),
        ],
    )
    def test_transform(self, array):
        a = dpp.LabelEncoder()
        b = spp.LabelEncoder()

        a.fit(array)
        b.fit(array.compute())

        assert_eq_ar(a.transform(array).compute(), b.transform(array.compute()))

    @pytest.mark.parametrize("array", [np_y, y, s])
    def test_transform_dtypes(self, array):
        result = dpp.LabelEncoder().fit_transform(array)
        assert result.dtype == np.intp
        if dask.is_dask_collection(array):
            assert result.dtype == result.compute().dtype

    def test_fit_transform_categorical(self):
        cat = dd.from_pandas(pd.Series(choices, dtype="category"), npartitions=4)
        result = dpp.LabelEncoder().fit_transform(cat)
        assert result.dtype == "int8"
        assert result.dtype == result.compute().dtype

    @pytest.mark.parametrize("array", [y, s])
    def test_inverse_transform(self, array):

        a = dpp.LabelEncoder()
        assert_eq_ar(a.inverse_transform(a.fit_transform(array)), da.asarray(array))

    @pytest.mark.parametrize(
        "categories, transformed",
        [
            (["a", "b", "c"], np.array([0, 1, 0], dtype=np.int8)),
            (["a", "b"], np.array([0, 1, 0], dtype=np.int8)),
            (["b", "a"], np.array([1, 0, 1], dtype=np.int8)),
        ],
    )
    @pytest.mark.parametrize("daskify", [True, False, "unknown"])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_categorical(self, categories, transformed, daskify, ordered):
        cat = pd.Series(
            ["a", "b", "a"],
            dtype=pd.api.types.CategoricalDtype(categories=categories, ordered=ordered),
        )
        if daskify:
            cat = dd.from_pandas(cat, npartitions=2)
            transformed = da.from_array(transformed, chunks=(2, 1))
            if daskify == "unknown":
                cat = cat.cat.as_unknown()

        a = dpp.LabelEncoder().fit(cat)

        if daskify != "unknown":
            assert a.dtype_ == cat.dtype
        np.testing.assert_array_equal(a.classes_, categories)
        result = a.transform(cat)
        da.utils.assert_eq(result, transformed)

        inv_transformed = a.inverse_transform(result)
        if daskify:
            # manually set the divisions for the test
            inv_transformed.divisions = (0, 2)
        dd.utils.assert_eq(inv_transformed, cat)

    def test_dataframe_raises(self):
        df = pd.DataFrame({"A": ["a", "a", "b"]}, dtype="category")
        dpp.LabelEncoder().fit(df)  # OK

        df["other"] = ["a", "b", "c"]
        with pytest.raises(ValueError):
            dpp.LabelEncoder().fit(df)

    @pytest.mark.parametrize("daskify", [True, False])
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.xfail(reason="Incorrect 32-bit dtype.", strict=False)
    def test_use_categorical(self, daskify):
        data = pd.Series(
            ["b", "c"], dtype=pd.api.types.CategoricalDtype(["c", "a", "b"])
        )
        if daskify:
            data = dd.from_pandas(data, npartitions=2)
        a = dpp.LabelEncoder(use_categorical=False).fit(data)
        b = spp.LabelEncoder().fit(data)
        assert_estimator_equal(a, b, exclude={"dtype_"})
        assert a.dtype_ is None

        da.utils.assert_eq(a.transform(data), b.transform(data))
        a_trn = a.transform(data)
        b_trn = b.transform(data)
        da.utils.assert_eq(a_trn, b_trn)
        da.utils.assert_eq(a.inverse_transform(a_trn), b.inverse_transform(b_trn))

    def test_unseen_raises_array(self):
        enc = dpp.LabelEncoder().fit(y)
        new = da.from_array(np.array(["a", "a", "z"]), chunks=2)
        result = enc.transform(new)

        with pytest.raises(ValueError):
            result.compute()
