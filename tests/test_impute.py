import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import sklearn.impute

import dask_ml.datasets
import dask_ml.impute
from dask_ml._compat import DASK_2_26_0
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


def test_simple_imputer_add_indicator_raises():
    # https://github.com/dask/dask-ml/issues/494
    pytest.importorskip("sklearn", minversion="0.21.dev0")
    imputer = dask_ml.impute.SimpleImputer(add_indicator=True)

    with pytest.raises(NotImplementedError):
        imputer.fit(dX)


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
    elif daskify and strategy == "median" and DASK_2_26_0:
        # New quantile implementation in Dask
        expected = pd.Series([1.0], index=["A"])
    else:
        expected = pd.Series([2], index=["A"])
    tm.assert_series_equal(b.statistics_, expected, check_dtype=False)


def test_impute_most_frequent():
    # https://github.com/dask/dask-ml/issues/385
    data = dd.from_pandas(pd.DataFrame([1, 1, 1, 1, np.nan, np.nan]), 2)
    model = dask_ml.impute.SimpleImputer(strategy="most_frequent")
    result = model.fit_transform(data)
    expected = dd.from_pandas(pd.DataFrame({0: [1.0] * 6}), 2)
    dd.utils.assert_eq(result, expected)
    assert model.statistics_[0] == 1.0
