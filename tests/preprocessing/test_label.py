import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from dask.array.utils import assert_eq as assert_eq_ar

import dask_ml.preprocessing as dpp
import sklearn.preprocessing as spp
from dask_ml.utils import assert_estimator_equal

choices = np.array(["a", "b", "c"], dtype=str)
y = np.random.choice(choices, 100)
y = da.from_array(y, chunks=13)
s = dd.from_array(y)


@pytest.fixture
def pandas_series():
    y = np.random.choice(["a", "b", "c"], 100)
    return pd.Series(y)


@pytest.fixture
def dask_array(pandas_series):
    return da.from_array(pandas_series, chunks=5)


class TestLabelEncoder(object):
    def test_basic(self):
        a = dpp.LabelEncoder()
        b = spp.LabelEncoder()

        a.fit(y)
        b.fit(y.compute())
        assert_estimator_equal(a, b)

    def test_input_types(self, dask_array, pandas_series):
        a = dpp.LabelEncoder()
        b = spp.LabelEncoder()

        assert_estimator_equal(a.fit(dask_array), b.fit(pandas_series))

        assert_estimator_equal(a.fit(pandas_series), b.fit(pandas_series))

        assert_estimator_equal(a.fit(pandas_series.values), b.fit(pandas_series))

        assert_estimator_equal(a.fit(dask_array), b.fit(pandas_series.values))

    @pytest.mark.parametrize("array", [y, s])
    def test_transform(self, array):
        a = dpp.LabelEncoder()
        b = spp.LabelEncoder()

        a.fit(array)
        b.fit(array.compute())

        assert_eq_ar(a.transform(array).compute(), b.transform(array.compute()))

    @pytest.mark.parametrize("array", [y, s])
    def test_inverse_transform(self, array):

        a = dpp.LabelEncoder()
        assert_eq_ar(a.inverse_transform(a.fit_transform(array)), da.asarray(array))
