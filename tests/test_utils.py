from collections import namedtuple
import pytest

from daskml.utils import slice_columns, handle_zeros_in_scale
from dask.array.utils import assert_eq as assert_eq_ar
from dask.dataframe.utils import assert_eq as assert_eq_df
from daskml.datasets import make_classification
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np


df = dd.from_pandas(pd.DataFrame(5*[range(42)]).T, npartitions=5)
s = dd.from_pandas(pd.Series([0, 1, 2, 3, 0]), npartitions=5)
a = da.from_array(np.array([0, 1, 2, 3, 0]), chunks=3)
X, y = make_classification(chunks=2)

Foo = namedtuple('Foo', 'a_ b_ c_ d_')
Bar = namedtuple("Bar", 'a_ b_ d_ e_')


def assert_estimator_equal(left, right, exclude=None, **kwargs):
    """Check that two Estimators are equal

    Parameters
    ----------
    left, right : Estimators
    exclude : str or sequence of str
        attributes to skip in the check
    kwargs : dict
        Passed through to the dask `assert_eq` method.

    """
    left_attrs = [x for x in dir(left) if x.endswith('_')
                  and not x.startswith('_')]
    right_attrs = [x for x in dir(right) if x.endswith('_')
                   and not x.startswith('_')]
    if exclude is None:
        exclude = set()
    elif isinstance(exclude, str):
        exclude = {exclude}
    else:
        exclude = set(exclude)

    assert (set(left_attrs) - exclude) == set(right_attrs) - exclude

    for attr in set(left_attrs) - exclude:
        l = getattr(left, attr)
        r = getattr(right, attr)
        if isinstance(l, (np.ndarray, da.Array)):
            assert_eq_ar(l, r, **kwargs)
        elif isinstance(l, (pd.core.generic.NDFrame, dd._Frame)):
            assert_eq_df(l, r, **kwargs)
        else:
            assert l == r


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
