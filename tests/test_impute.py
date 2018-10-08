import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pytest
import sklearn.impute

import dask_ml.datasets
import dask_ml.impute
from dask_ml.utils import assert_estimator_equal

rng = np.random.RandomState(0)

X = rng.uniform(size=(10, 4))
X[X < 0.5] = np.nan

dX = da.from_array(X, chunks=5)
df = pd.DataFrame(X)
ddf = dd.from_pandas(df, npartitions=2)


@pytest.mark.parametrize("data", [X, dX, df, ddf])
def test_fit(data):
    a = sklearn.impute.SimpleImputer()
    b = dask_ml.impute.SimpleImputer()

    a.fit(X)
    b.fit(data)

    assert_estimator_equal(a, b)


@pytest.mark.parametrize("data", [X, dX, df, ddf])
def test_fit_constant(data):
    a = sklearn.impute.SimpleImputer(strategy="constant", fill_value=-999.0)
    b = dask_ml.impute.SimpleImputer(strategy="constant", fill_value=-999.0)

    expected = a.fit_transform(X)
    result = b.fit_transform(data)

    assert_estimator_equal(a, b)
    assert isinstance(result, type(data))
    if isinstance(data, (pd.DataFrame, dd.DataFrame)):
        result = result.values

    da.utils.assert_eq(result, expected)


@pytest.mark.parametrize("data", [X, dX, df, ddf])
def test_transform(data):
    a = sklearn.impute.SimpleImputer()
    b = dask_ml.impute.SimpleImputer()

    expected = a.fit_transform(X)
    result = b.fit_transform(data)

    assert isinstance(result, type(data))
    if isinstance(data, (pd.DataFrame, dd.DataFrame)):
        result = result.values

    da.utils.assert_eq(result, expected)


def test_invalid_raises():
    imp = dask_ml.impute.SimpleImputer(strategy="other")

    with pytest.raises(ValueError, match="other"):
        imp.fit(dX)


def test_invalid_missing_values():
    imp = dask_ml.impute.SimpleImputer(missing_values="foo")

    with pytest.raises(ValueError, match="non-NA values"):
        imp.fit(dX)


def test_array_median_raises():
    imp = dask_ml.impute.SimpleImputer(strategy="median")

    with pytest.raises(ValueError, match="Can only use"):
        imp.fit(dX)


@pytest.mark.parametrize("daskify", [True, False])
@pytest.mark.parametrize("strategy", ["median", "most_frequent", "constant"])
def test_frame_strategies(daskify, strategy):
    df = pd.DataFrame({"A": [1, 1, np.nan, np.nan, 2, 2]})
    if daskify:
        df = dd.from_pandas(df, 2)

    if strategy == "constant":
        fill_value = 2
    else:
        fill_value = None

    b = dask_ml.impute.SimpleImputer(strategy=strategy, fill_value=fill_value)
    b.fit(df)
    if not daskify and strategy == "median":
        expected = pd.Series([1.5], index=["A"])
    else:
        expected = pd.Series([2], index=["A"])
    tm.assert_series_equal(b.statistics_, expected, check_dtype=False)
