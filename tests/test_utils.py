from collections import namedtuple

import pytest
import pandas as pd
import pandas.util.testing as tm
import six
import numpy as np
import dask.dataframe as dd
import dask.array as da
from dask.array.utils import assert_eq as assert_eq_ar
from dask.dataframe.utils import assert_eq as assert_eq_df

from dask_ml.utils import (
    slice_columns, handle_zeros_in_scale, assert_estimator_equal,
    check_random_state,
    check_chunks,
    check_array,
)
from dask_ml.datasets import make_classification


df = dd.from_pandas(pd.DataFrame(5 * [range(42)]).T, npartitions=5)
s = dd.from_pandas(pd.Series([0, 1, 2, 3, 0]), npartitions=5)
a = da.from_array(np.array([0, 1, 2, 3, 0]), chunks=3)
X, y = make_classification(chunks=(2, 20))

Foo = namedtuple('Foo', 'a_ b_ c_ d_')
Bar = namedtuple("Bar", 'a_ b_ d_ e_')


def test_slice_columns():
    columns = [2, 3]
    df2 = slice_columns(df, columns)
    X2 = slice_columns(X, columns)

    assert list(df2.columns) == columns
    assert_eq_df(df[columns].compute(), df2.compute())
    assert_eq_ar(X.compute(), X2.compute())


def test_handle_zeros_in_scale():
    s2 = handle_zeros_in_scale(s)
    a2 = handle_zeros_in_scale(a)

    assert list(s2.compute()) == [1, 1, 2, 3, 1]
    assert list(a2.compute()) == [1, 1, 2, 3, 1]

    x = np.array([1, 2, 3, 0], dtype='f8')
    expected = np.array([1, 2, 3, 1], dtype='f8')
    result = handle_zeros_in_scale(x)
    np.testing.assert_array_equal(result, expected)

    x = pd.Series(x)
    expected = pd.Series(expected)
    result = handle_zeros_in_scale(x)
    tm.assert_series_equal(result, expected)

    x = da.from_array(x.values, chunks=2)
    expected = expected.values
    result = handle_zeros_in_scale(x)
    assert_eq_ar(result, expected)

    x = dd.from_dask_array(x)
    expected = pd.Series(expected)
    result = handle_zeros_in_scale(x)
    assert_eq_df(result, expected)


def test_assert_estimator_passes():
    l = Foo(1, 2, 3, 4)
    r = Foo(1, 2, 3, 4)
    assert_estimator_equal(l, r)  # it works!


def test_assert_estimator_different_attributes():
    l = Foo(1, 2, 3, 4)
    r = Bar(1, 2, 3, 4)
    with pytest.raises(AssertionError):
        assert_estimator_equal(l, r)


def test_assert_estimator_different_scalers():
    l = Foo(1, 2, 3, 4)
    r = Foo(1, 2, 3, 3)
    with pytest.raises(AssertionError):
        assert_estimator_equal(l, r)


@pytest.mark.parametrize('a', [
    np.array([1, 2]),
    da.from_array(np.array([1, 2]), chunks=1),
])
def test_assert_estimator_different_arrays(a):
    l = Foo(1, 2, 3, a)
    r = Foo(1, 2, 3, np.array([1, 0]))
    with pytest.raises(AssertionError):
        assert_estimator_equal(l, r)


@pytest.mark.parametrize('a', [
    pd.DataFrame({"A": [1, 2]}),
    dd.from_pandas(pd.DataFrame({"A": [1, 2]}), npartitions=2),
])
def test_assert_estimator_different_dataframes(a):
    l = Foo(1, 2, 3, a)
    r = Foo(1, 2, 3, pd.DataFrame({"A": [0, 1]}))
    with pytest.raises(AssertionError):
        assert_estimator_equal(l, r)


def test_check_random_state():
    for rs in [None, 0]:
        result = check_random_state(rs)
        assert isinstance(result, da.random.RandomState)

    rs = da.random.RandomState(0)
    result = check_random_state(rs)
    assert result is rs

    with pytest.raises(TypeError):
        check_random_state(np.random.RandomState(0))


@pytest.mark.parametrize('chunks', [
    None, 4, (2000, 4), [2000, 4],
])
@pytest.mark.skipif(six.PY2, reason="No mock")
def test_get_chunks(chunks):
    from unittest import mock

    with mock.patch("dask_ml.utils.cpu_count", return_value=4):
        result = check_chunks(n_samples=8000, n_features=4, chunks=chunks)
        expected = (2000, 4)
        assert result == expected


@pytest.mark.parametrize('chunks', [None, 8])
def test_get_chunks_min(chunks):
    result = check_chunks(n_samples=8, n_features=4, chunks=chunks)
    expected = (100, 4)
    assert result == expected


def test_get_chunks_raises():
    with pytest.raises(AssertionError):
        check_chunks(1, 1, chunks=(1, 2, 3))

    with pytest.raises(AssertionError):
        check_chunks(1, 1, chunks=[1, 2, 3])

    with pytest.raises(ValueError):
        check_chunks(1, 1, chunks=object())


def test_check_array_raises():
    X = da.random.uniform(size=(10, 5), chunks=2)
    with pytest.raises(TypeError) as m:
        check_array(X)

    assert m.match("Chunking is only allowed on the first axis.")
